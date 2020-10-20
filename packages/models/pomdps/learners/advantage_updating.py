from abc import ABC, abstractmethod
from typing import Optional, Any, Union

import torch
from torch import nn as nn
from torch.optim.optimizer import Optimizer

from packages.models.components.advantages import AdvantageNet
from packages.models.components.values import ValueNet
from packages.models.components.observations import ContinuousObservationModel, RandomDiscreteDensityObservationModel
from packages.models.components.filters import Filter, ContinuousRandomDiscreteFilter, WonhamFilter
from packages.models.pomdps.learners.learner import Learner
from packages.models.pomdps.models import POMDP
from packages.models.pomdps.policys import AdvantagePolicy
from packages.types.spaces import FiniteIntSpace
from collections import namedtuple
import math
from scipy.misc import derivative
from torch.nn.functional import smooth_l1_loss, mse_loss
from packages.utils.utils import ReplayMemory
from packages.models.pomdps.simulator import ObservedData, POMDPSimulator


class AdvantageUpdateLearner(Learner):
    """ Advantage updating for the POMDP. Finds Approximate Solution using function approximation """

    def __init__(self, pomdp: POMDP, name: str = "au", num_episodes: int = 1000,
                 opt_constructor: Optional[Optimizer] = None, optimizer_v_options=None, optimizer_a_options=None,
                 advantage_net: AdvantageNet = None, value_net: ValueNet = None,
                 checkpoint_interval: int = 50,
                 batch_size: int = 128,
                 episode_length: Optional[Union[Any, torch.Tensor]] = 100,
                 reset_episode: bool = True,
                 num_optim_iter: int = 100,
                 filter: Optional[Filter] = None, ode_options=None, exploration_options=None,
                 num_episode_samples: int = 100,
                 memory_capacity: int = int(1e6),
                 discount_decay: int = 0,
                 sample_initial_belief=None):
        """

        :param pomdp: The POMDP to be solved
        :param name: Name of the learning process used for logging and checkpoints
        :param num_episodes:  Number of episodes used for advantage updating
        :param opt_constructor: Optimizer used for network training
        :param optimizer_v_options: Options passed on to the value function optimizer
        :param optimizer_a_options: Options passed on to the advantage function optimizer
        :param advantage_net: NN representing the advantage function
        :param value_net: NN representing the value function
        :param checkpoint_interval: Interval of epochs for saving checkpoint models. 0 will disable checkpoints.
        :param batch_size: Number of samples taken from the replay buffer
        :param episode_length: time length of one episode
        :param reset_episode: Boolean if true the episode starts from the initial belief otherwise belief is attained
        :param num_optim_iter: Number of gradient steps per episode
        :param filter: Filter for the POMDP
        :param ode_options:  Options passed on to the ODE solvers
        :param exploration_options:  Options passed to advantage_policy.ou_pertubation
        :param num_episode_samples: Number of subsamples used for one trajectory
        :param memory_capacity: Number of samples in the cyclic replay buffer
        :param discount_decay: Decay for the discount parameter. 0 disables decay
        """
        super().__init__(pomdp=pomdp, name=name, num_episodes=num_episodes,
                         opt_constructor=opt_constructor,
                         optimizer_v_options=optimizer_v_options, optimizer_a_options=optimizer_a_options,
                         advantage_net=advantage_net, value_net=value_net,
                         checkpoint_interval=checkpoint_interval,
                         discount_decay=discount_decay)
        # Device used in pomdp
        device = pomdp.discount.device

        # --------------- cast inputs ---------------
        assert batch_size > 0
        self.batch_size = batch_size
        self.register_buffer('episode_length', torch.as_tensor(episode_length, device=device))
        self.num_optim_iter = num_optim_iter
        self.ode_options = ode_options
        self.policy = AdvantagePolicy(pomdp.SSpace, pomdp.ASpace, self.advantage_net)
        self.start_state = None  # Start with default start state
        self.reset_episode = reset_episode

        # --------------- initial belief ---------------
        if sample_initial_belief is None:
            initial_belief = torch.ones(self.pomdp.SSpace.nElements, dtype=torch.float)
            initial_belief /= initial_belief.sum()
            self.register_buffer('initial_belief', initial_belief)
            self.fun_sample_initial_belief = None
        else:
            self.initial_belief = None
            if callable(sample_initial_belief):
                self.fun_sample_initial_belief = sample_initial_belief
            else:
                belief_dist = torch.distributions.dirichlet.Dirichlet(sample_initial_belief)
                self.fun_sample_initial_belief = lambda: belief_dist.rsample().to(device)

        # --------------- filter ---------------
        if filter is None:
            # Set default filter
            if isinstance(pomdp.OModel, RandomDiscreteDensityObservationModel):
                self.filter = ContinuousRandomDiscreteFilter(pomdp.SSpace,
                                                             pomdp.ASpace,
                                                             pomdp.TModel,
                                                             pomdp.OModel,
                                                             self.policy)
            elif isinstance(pomdp.OModel, ContinuousObservationModel):
                self.filter = WonhamFilter(pomdp.SSpace,
                                           pomdp.ASpace,
                                           pomdp.TModel,
                                           pomdp.OModel,
                                           self.policy)
            else:
                raise TypeError('No default filter implemented for observation model')

        else:
            self.filter = filter

        # --------------- exploration options ---------------
        if exploration_options is None:
            exploration_options = dict()

        try:
            exploration_options['dt'] = torch.as_tensor(exploration_options['dt'], device=device)
        except KeyError:
            assert isinstance(pomdp.SSpace, FiniteIntSpace)
            exploration_options['dt'] = torch.tensor(1e-2, device=device)
            # 1 / ( 10 * pomdp.TModel.max_rate(torch.arange(pomdp.SSpace.nElements, device=device)).max())

        try:
            exploration_options['kappa'] = torch.as_tensor(exploration_options['kappa'], device=device)
        except KeyError:
            exploration_options['kappa'] = None

        try:
            exploration_options['sigma_start']
        except KeyError:
            exploration_options['sigma_start'] = 1.5
        try:
            exploration_options['sigma_end']
        except KeyError:
            exploration_options['sigma_end'] = .5
        try:
            exploration_options['sigma_decay']
        except KeyError:
            exploration_options['sigma_decay'] = 100
        self.exploration_options = exploration_options

        # --------------- loss ---------------
        self.memory_capacity = memory_capacity
        self.num_episode_samples = num_episode_samples
        if isinstance(self.filter, ContinuousRandomDiscreteFilter):
            self.loss_module = PiecewiseDeterministicMarkovLoss(self.batch_size, self.num_episode_samples,
                                                                self.episode_length, self.value_net, self.advantage_net,
                                                                self.pomdp.discount, self.memory_capacity,
                                                                self.pomdp.OModel.observationRate)
        elif isinstance(self.filter, WonhamFilter):
            self.loss_module = DiffusionLoss(self.batch_size, self.num_episode_samples,
                                             self.episode_length, self.value_net, self.advantage_net,
                                             self.pomdp.discount, self.memory_capacity)
        else:
            raise NotImplementedError('Unkown filter')

        self.sim = POMDPSimulator(self.pomdp)

        # Send everything to same device as the pomdp
        self.to(device)

    def train_episode(self, episode):
        if self.initial_belief is None:
            initial_belief = self.fun_sample_initial_belief().to(self.pomdp.discount.device)
            start_state = torch.argmax(initial_belief)
        else:
            initial_belief = self.initial_belief
            start_state = self.start_state

        loss = 0

        self.advantage_net.requires_grad_(False)
        with torch.no_grad():

            # Use an exploration decay on the dispersion of the ou pertubation
            sigma_decay = self.exploration_options['sigma_decay']
            if sigma_decay > 0:
                sigma = self.exploration_options['sigma_end'] + (
                        self.exploration_options['sigma_start'] - self.exploration_options['sigma_end']) * math.exp(
                        -1. * episode / sigma_decay)
            else:
                sigma = self.exploration_options['sigma_end']

            self.policy.ou_perturbation(self.episode_length,
                                        self.exploration_options['dt'],
                                        kappa=self.exploration_options['kappa'],
                                        sigma=sigma)  # Add exploration to policy
            # Sample a trajectory
            # t_grid = torch.linspace(0, self.episode_length, self.num_episode_samples, device=self.episode_length.device)
            # Random grid
            t_grid = torch.unique(
                torch.rand(self.num_episode_samples, device=self.episode_length.device)* self.episode_length)

            latent_traj, observed_traj, _ = self.sim.sampleTraj(t_grid, self.policy, self.filter,
                                                                initial_belief,
                                                                start_state=start_state,
                                                                ode_options=self.ode_options)
            self.logger.info('Sampled trajectory for  - episode: %d' % episode)

        # decay the discount factor
        self.loss_module.gamma = -1 / self.get_decayed_log_discount(episode)

        # Add the trajectory to the loss
        self.loss_module.add_traj(observed_traj)

        # Check if belief and state should be attained after an episode
        if not self.reset_episode:
            self.initial_beliefs = observed_traj[-1].belief
            self.start_states = latent_traj[-1].state

        # Optimize advantage and value net
        self.advantage_net.requires_grad_(True)
        for optim_iter in range(self.num_optim_iter):
            # Value update
            loss = self.loss_module.sample()
            self.optimizer_v.zero_grad()
            self.optimizer_a.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer_v.step()
            self.optimizer_a.step()
            loss = loss.item()

        # data = None
        # TODO
        data = self.loss_module.memory_trajs.memory

        self.logger.info(
                'episode: %d - optim_iter: %d - loss: %f' % (episode, self.num_optim_iter, loss))
        return data

    def get_parameters(self):
        params = super().get_parameters()
        params['batch_size'] = self.batch_size
        params['episode_length'] = self.episode_length
        params['sigma_decay'] = self.exploration_options['sigma_decay']
        params['discount_decay'] = self.discount_decay
        params['num_opt_iter'] = self.num_optim_iter
        params['memory_capacity'] = self.memory_capacity
        params['reset_episode'] = self.reset_episode
        params['num_episode_samples'] = self.num_episode_samples
        return params


Traj_Element = namedtuple('Traj_Element', ('belief', 'belief_derivative', 'action', 'reward'))


class Loss(ABC, nn.Module):
    def __init__(self, batch_size: int, num_episode_samples: int, episode_length, value_net, advantage_net, discount):
        """
        Base class for a loss to be optimized
        :param batch_size: Number of samples from the replay buffer used per optimization iteration
        :param num_episode_samples: Number of samples used for subsampling the trajectory
        :param episode_length: Length of one episode
        :param advantage_net: NN representing the advantage function
        :param value_net: NN representing the value function
        :param discount: discount parameter of the pomdp
        """

        # -------------- Save inputs --------------
        nn.Module.__init__(self)
        self.batch_size = batch_size
        self.num_episode_samples = num_episode_samples
        self.value_net = value_net
        self.advantage_net = advantage_net
        self.register_buffer('episode_length', episode_length)
        self.register_buffer('log_discount', discount.log())
        self.register_buffer('gamma', -1 / discount.log())

    @abstractmethod
    def sample(self):
        """
        Base class sampling a batch of samples from the replay buffer and returns the loss
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def add_traj(self, observed_traj):
        """
        Subsamples a trajectory and saves it to the replay buffer
        :param observed_traj: A Trajectory object
        """
        raise NotImplementedError()


# TODO Not tested
class DiffusionLoss(Loss):
    def __init__(self, batch_size, num_episode_samples, episode_length, value_net, advantage_net, discount,
                 memory_capacity):
        super().__init__(batch_size, num_episode_samples, episode_length, value_net, advantage_net, discount)
        self.memory_trajs = ReplayMemory(memory_capacity)

    def sample(self):
        # -----------  trajectory loss -----------
        n_traj = min(self.batch_size, len(self.memory_trajs))  # If not enough elements in buffer sample all from memory
        traj_elements = self.memory_trajs.sample(n_traj)
        traj_batch = ObservedData(*zip(*traj_elements))
        beliefs = torch.stack(traj_batch.belief)
        belief_derivatives = torch.stack(traj_batch.belief_derivative)
        actions = torch.stack(traj_batch.action)
        rewards = torch.stack(traj_batch.reward)

        value, value_grad, value_hessian = self.value_net(beliefs, compute_grad=True, compute_hessian=True)
        advantage = self.advantage_net(beliefs)

        loss = mse_loss((self.gamma * advantage.gather(1, actions[:, None]).squeeze(dim=1)
                         + value.squeeze(dim=1)
                         - self.gamma * torch.sum(belief_derivatives * value_grad, dim=1)
                         - self.gamma * 1 / 2 * torch.sum(
                        value_hessian * belief_derivatives @ belief_derivatives.permute(0, 2, 1), dim=(1, 2))
                         ), rewards.squeeze(dim=1))
        return loss

    def add_traj(self, observed_traj):
        for i in range(len(observed_traj)):
            self.memory_trajs.push(observed_traj[i])


class PiecewiseDeterministicMarkovLoss(Loss):
    def __init__(self, batch_size, num_episode_samples, episode_length, value_net, advantage_net, discount,
                 memory_capacity, observationRate):
        """
        Loss for the continuous discrete filter
        :param memory_capacity: size of the replay buffer
        """
        super().__init__(batch_size, num_episode_samples, episode_length, value_net, advantage_net, discount)
        self.memory_trajs = ReplayMemory(memory_capacity)
        self.register_buffer('observationRate', observationRate)

    def sample(self):
        # -----------  trajectory loss -----------
        n_traj = min(self.batch_size, len(self.memory_trajs))  # If not enough elements in buffer sample all from memory
        traj_elements = self.memory_trajs.sample(n_traj)
        traj_batch = ObservedData(*zip(*traj_elements))
        beliefs = torch.stack(traj_batch.belief)
        beliefs_plus = torch.stack(traj_batch.belief_plus)
        belief_derivatives = torch.stack(traj_batch.belief_derivative)
        actions = torch.stack(traj_batch.action)
        rewards = torch.stack(traj_batch.reward)

        value, value_grad = self.value_net(beliefs, compute_grad=True, compute_hessian=False)
        value_plus = self.value_net(beliefs_plus, compute_grad=False, compute_hessian=False)
        advantage = self.advantage_net(beliefs)
        loss = mse_loss((advantage.gather(1, actions[:, None]).squeeze(dim=1)
                         + value.squeeze(dim=1)
                         - self.gamma * torch.sum(belief_derivatives * value_grad, dim=1)
                         - self.gamma * self.observationRate * (value_plus.squeeze(dim=1) - value.squeeze(dim=1))),
                        rewards.squeeze(dim=1))
        return loss

    def add_traj(self, observed_traj):
        # Subsample the trajectory on a grid and add the elements
        # t = torch.linspace(0, self.episode_length, self.num_episode_samples, device=self.episode_length.device)
        # t = torch.rand(self.num_episode_samples, device=self.episode_length.device) * self.episode_length
        for i in range(len(observed_traj)):
            self.memory_trajs.push(observed_traj[i])
