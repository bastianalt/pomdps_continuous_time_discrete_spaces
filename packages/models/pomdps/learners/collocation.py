import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.distributions import Categorical
from typing import Optional

from packages.models.components.dynamics import WonhamDiffusion
from packages.models.components.rewards import TabularRewardModel
from packages.models.pomdps.learners.learner import Learner
from packages.models.pomdps.models import POMDP
from packages.types.spaces import FiniteIntSpace
from packages.models.components.advantages import AdvantageNet
from packages.models.components.values import ValueNet
from packages.models.components.transitions import TabularTransitionModel
from packages.models.components.observations import RandomDiscreteDensityObservationModel, ContinuousObservationModel, \
    RandomFiniteObservationModel
import time
import math


class CollocationLearner(Learner):
    """ This class is responsible for learning the policy of a POMDP using the collocation method """

    def __init__(self, pomdp, name: str, num_episodes: int, opt_constructor: Optional[Optimizer] = None,
                 optimizer_v_options=None, optimizer_a_options=None, advantage_net: AdvantageNet = None,
                 value_net: ValueNet = None,
                 checkpoint_interval: int = 50, belief_prior=None, batch_size=100, num_iter_v: int = 10,
                 num_iter_a: int = 10, approx_state_exp_sampling=0, discount_decay: int = 0, **kwargs):
        """
        :param pomdp: The POMDP to be solved
        :param name: Name of the learning process used for logging and checkpoints
        :param num_episodes: Number of episodes to train for
        :param opt_constructor: Optimizer used for network training
        :param optimizer_v_options: Options for the optimizer used in network training
        :param advantage_net: The advantage network used to represent the advantage function
        :param value_net: The value network used to represent the value function
        :param checkpoint_interval: Interval of epochs for saving checkpoint models. 0 will disable checkpoints.
        :param belief_prior: The prior (parameters for the Dirichlet distribution) that beliefs are sampled from.
                             Must be a tensor of size of the number of states.
        :param batch_size: Batch size for sampled belief states used within one iteration
        :param num_iter_v: Number of iterations to train value network for within one episode
        :param num_iter_a: Number of iterations to train advantage network for within one episode
        :param approx_state_exp_sampling: Whether state expectations should be approximated using a single sample.
                                          0 to approximate nothing (only for discrete finite observations)
                                          1 to approximate observation expectation in jump term
                                          2 to approximate observation and state expectation in jump term
                                          3 to approximate observation and state expectation in jump and transition term
        """
        super().__init__(pomdp=pomdp, name=name, num_episodes=num_episodes,
                         opt_constructor=opt_constructor,
                         optimizer_v_options=optimizer_v_options, optimizer_a_options=optimizer_a_options,
                         advantage_net=advantage_net, value_net=value_net,
                         checkpoint_interval=checkpoint_interval, discount_decay=discount_decay)

        self.batch_size = batch_size
        self.num_iter_v = num_iter_v
        self.num_iter_a = num_iter_a
        self.sample_states_hjb = approx_state_exp_sampling

        if belief_prior is None:
            belief_prior = torch.ones(self.pomdp.SSpace.nElements)
        # self.register_buffer('belief_prior', belief_prior)

        # create model for sampling and training that facilitates computing gradients efficiently
        if isinstance(self.pomdp.OModel, RandomDiscreteDensityObservationModel):
            self.hjb = DiscreteObservationCollocationHJBModel(self.pomdp, self.batch_size, self.value_net,
                                                              belief_prior,
                                                              approx_state_exp_sampling=self.sample_states_hjb,
                                                              **kwargs)
        elif isinstance(self.pomdp.OModel, ContinuousObservationModel):
            self.hjb = WonhamCollocationHJBModel(self.pomdp, self.batch_size, self.value_net, belief_prior,
                                                 approx_state_exp_sampling=self.sample_states_hjb > 1, **kwargs)
        else:
            raise TypeError("Observation model is neither a discrete nor a continuous process.")
        self.to(pomdp.discount)

    def train_episode(self, episode):
        # decay discount
        self.hjb.log_discount = self.get_decayed_log_discount(episode)
        # sample a new batch of data points for training
        self.hjb.resample()

        # update value network
        val_loss = 0
        for i in range(self.num_iter_v):
            #print(i)
            self.optimizer_v.zero_grad()

            loss, _, advals_a = self.hjb()
            loss.backward()

            self.optimizer_v.step()
            val_loss += loss.item()
            # self.logger.info("opt val net ep. {}, it {}, loss {}".format(episode + 1, iter + 1, loss))

        mean_v_loss = val_loss / self.num_iter_v if self.num_iter_v > 0 else float('nan')
        self.logger.info("Val loss {}".format(mean_v_loss))

        return None

    def learn_advantage(self):
        """
        Learns the advantage function for the current value net
        """
        self.running_loss = 0

        for e in range(self.num_episodes):
            # measure time for statistics
            start = time.time()

            self.hjb.resample()
            # compute advantage values again using updated params
            with torch.no_grad():
                _, _, advals_a = self.hjb()

            # update advantage network
            adv_loss = 0
            for i in range(self.num_iter_a):
                self.optimizer_a.zero_grad()

                loss = self._advantage_loss(self.advantage_net, self.hjb.batch_beliefs, advals_a)
                loss.backward()

                self.optimizer_a.step()
                adv_loss += loss.item()
                # logger.info("opt adv net ep. {}, it {}, loss {}".format(e + 1, iter + 1, loss))

            mean_a_loss = adv_loss / self.num_iter_a if self.num_iter_a > 0 else float('nan')
            self.logger.info("Adv loss {}".format(mean_a_loss))
            loss = mean_a_loss

            self.running_loss += loss

            # save checkpoints
            if self.checkpoint_interval > 0 and (e % self.checkpoint_interval == 0 or e == self.num_episodes - 1):
                torch.save({
                    'value_net_state_dict': self.value_net.state_dict(),
                    'advantage_net_state_dict': self.advantage_net.state_dict(),
                    'pomdp_params': self.pomdp.get_parameters(),
                    'learner_params': self.get_parameters(),
                    'episode': e,
                }, "checkpoints/model_col_adv_{}_{}".format(self.name, e))

            end = time.time()
            self.logger.info("Episode {} took {}s.".format(e, end - start))

    def get_parameters(self):
        params = super().get_parameters()
        params['batch_size'] = self.batch_size
        params['num_iter_v'] = self.num_iter_v
        params['num_iter_a'] = self.num_iter_a
        params['approx_state_exp_sampling'] = self.sample_states_hjb
        params['ignore_jump_terms'] = self.hjb.ignore_jump_terms
        return params

    def _advantage_loss(self, advantage_net: AdvantageNet, beliefs: torch.Tensor, advals_a: torch.Tensor):
        """
        Computes the loss of the advantage network
        :param advantage_net: The advantage net to compute the loss for
        :param beliefs: The sampled beliefs
        :param advals_a: The adavantage values for all actions corresponding to the sampled beliefs
        :return: The loss of the advantage net w.r.t. the advantage values
        """
        aphi = advantage_net(beliefs)
        res = aphi - advals_a
        loss = torch.pow(res, 2).sum()
        return loss


class CollocationHJBModel(nn.Module):
    """
    This module creates a batch of collocation samples and computes the advantage values and the value network loss for
    collocation learners
    """

    def __init__(self, pomdp: POMDP, batch_size: int, value_net: ValueNet, belief_prior=None, belief_sample_fn=None,
                 approx_state_exp_sampling=0, ignore_jump_terms=False, **kwargs) -> None:
        super().__init__(**kwargs)

        assert (isinstance(pomdp.SSpace, FiniteIntSpace))
        assert (isinstance(pomdp.TModel, TabularTransitionModel))

        # set parameters
        self.pomdp = pomdp
        self.batch_size = batch_size
        self.approx_state_exp_sampling = approx_state_exp_sampling
        self.ignore_jump_terms = ignore_jump_terms

        self.n_states = pomdp.SSpace.nElements
        self.n_actions = pomdp.ASpace.nElements

        self.value_net = value_net

        if belief_prior is None:
            belief_prior = torch.ones(self.pomdp.SSpace.nElements)
        # distribution for sampling beliefs
        self.belief_dist = torch.distributions.dirichlet.Dirichlet(belief_prior)

        self.belief_sample_fn = belief_sample_fn

        if isinstance(self.pomdp.RModel, TabularRewardModel):
            reward_matrix = self.pomdp.RModel.rewardMatrix
        else:
            raise NotImplementedError()

        self.register_buffer('reward_matrix', reward_matrix)
        self.register_buffer('tm', pomdp.TModel.transitionMatrix)
        self.register_buffer('log_discount', torch.as_tensor(pomdp.discount).log())
        self.register_buffer('reward_matrix', reward_matrix)

        # parameters to store sampled batch and precomputed quantities for batch
        self.batch_beliefs = None
        self.batch_rewards = None

    def resample(self):
        """
        Renews the collocation sample batch
        """

        with torch.no_grad():
            # sample a batch of beliefs
            if self.belief_sample_fn is None:
                self.batch_beliefs = self.belief_dist.rsample([self.batch_size]).to(self.device)
            else:
                self.batch_beliefs = self.belief_sample_fn(self.batch_size).to(self.device)

            # Rewards for batch N, A
            self.batch_rewards = self.batch_beliefs @ self.reward_matrix

    @property
    def device(self):
        return self.log_discount.device


class DiscreteObservationCollocationHJBModel(CollocationHJBModel):

    def __init__(self, pomdp: POMDP, batch_size: int, value_net: ValueNet, belief_prior: torch.Tensor = None,
                 approx_state_exp_sampling=1, **kwargs) -> None:
        super().__init__(pomdp=pomdp, batch_size=batch_size, value_net=value_net, belief_prior=belief_prior,
                         approx_state_exp_sampling=approx_state_exp_sampling, **kwargs)

        self.mu = self.pomdp.OModel.observationRate

        # parameters to store sampled batch and precomputed quantities for batch
        self.batch_rcs = None
        self.batch_g_term = None

    def resample(self):
        super().resample()

        with torch.no_grad():
            actions_all = torch.arange(self.n_actions, device=self.device)
            states_all = torch.arange(self.n_states, device=self.device)

            if self.approx_state_exp_sampling >= 1:  # approximate expectations of term after \mu...
                # sample a state for each belief
                states = torch.distributions.Categorical(self.batch_beliefs).sample()

                # sample observations, shape N, A
                s_ba = states[:, None].repeat(1, self.n_actions)
                a_ba = actions_all[None, :].repeat(self.batch_size, 1)
                obs = self.pomdp.OModel.draw(s_ba.flatten(), a_ba.flatten()).reshape(self.batch_size, self.n_actions)

                # compute p(o given s, a) for all s. Shape N, A, S
                obs_nas = obs[:, :, None].repeat(1, 1, self.n_states)
                a_nas = actions_all[None, :, None].repeat(self.batch_size, 1, self.n_states)
                s_nas = states_all[None, None, :].repeat(self.batch_size, self.n_actions, 1)
                po = self.pomdp.OModel(obs_nas.flatten(), s_nas.flatten(), a_nas.flatten()).reshape(self.batch_size,
                                                                                                    self.n_actions,
                                                                                                    self.n_states).exp()

                # compute reset conditions for all s. Shape N, A, S
                numerator = po * self.batch_beliefs[:, None, :]
                self.batch_rcs = numerator / torch.sum(numerator, dim=-1, keepdim=True)

            elif self.approx_state_exp_sampling == 1:
                # sample observations for all batch element, state, and action combination. Shape N, A, S
                s_ba = states_all[None, None, :].repeat(self.batch_size, self.n_actions, 1)
                a_ba = actions_all[None, :, None].repeat(self.batch_size, 1, self.n_states)
                obs = self.pomdp.OModel.draw(s_ba.flatten(), a_ba.flatten()).reshape(self.batch_size, self.n_actions,
                                                                                     self.n_states)

                # compute p(o given s, a) for all s and s'. Shape N, A, S', S
                obs_nass = obs[..., None].repeat(1, 1, 1, self.n_states)  # N, A, S', S
                a_nass = actions_all[None, :, None, None].repeat(self.batch_size, 1, self.n_states, self.n_states)
                s_nass = states_all[None, None, None, :].repeat(self.batch_size, self.n_actions, self.n_states, 1)
                po = self.pomdp.OModel(obs_nass.flatten(), s_nass.flatten(), a_nass.flatten()).reshape(self.batch_size,
                                                                                                       self.n_actions,
                                                                                                       self.n_states,
                                                                                                       self.n_states).exp()

                # compute reset conditions for all s and s'. Shape N, A, S', S
                numerator = po * self.batch_beliefs[:, None, None, :]
                self.batch_rcs = numerator / torch.sum(numerator, dim=-1, keepdim=True)

            else:
                if not isinstance(self.pomdp.OModel, RandomFiniteObservationModel):
                    raise TypeError("Observation model must be finite discrete.")

                po = self.pomdp.OModel.observation_tensor.permute(0, 2, 1)  # shape O, A, S
                numerator = po[None, ...] * self.batch_beliefs[:, None, None, :]
                denumerator = torch.sum(numerator, dim=-1, keepdim=True)
                denumerator = torch.where(denumerator > 0, denumerator, torch.ones(1, device=self.device))  # set 0 in denominator to 1 as they will be multiplied with 0 anyways
                self.batch_rcs = numerator / denumerator  # shape N, O, A, S

            if self.approx_state_exp_sampling == 3:  # approximate transition term state expectation
                self.batch_g_term = self.tm[:, states, :].permute(1, 2, 0)  # shape N, A, S
            else:
                self.batch_g_term = torch.sum(self.batch_beliefs[:, None, :, None] * self.tm[None, ...],
                                              dim=2).permute(0, 2, 1)  # shape N, A, S
                # self.batch_g_term = torch.tensordot(self.batch_beliefs, self.tm, dims=([1], [0])).permute(0, 2, 1)

    def forward(self):
        """
        Computes the advantage values and their loss of the value network
        :return: The loss, advantage values, advantage values for all actions
        """
        vf, vg = self.value_net(self.batch_beliefs)

        # advantage values for all actions
        advals = self.batch_rewards - vf
        advals += -torch.sum(self.batch_g_term * vg[:, None, :], dim=-1)/self.log_discount

        # (un)comment to add/remove jump terms
        if not self.ignore_jump_terms:
            if self.batch_rcs.ndim == 3:
                vf_j = self.value_net(self.batch_rcs, compute_grad=False)[..., 0]
                advals += -self.mu * (vf_j - vf)/self.log_discount
            elif self.batch_rcs.ndim == 4:
                vf_j = self.value_net(self.batch_rcs, compute_grad=False)[..., 0]  # shape N, O, A
                if self.approx_state_exp_sampling == 1:
                    evf_j = (vf_j @ self.batch_beliefs[:, :, None])[..., 0]  # shape N, A
                elif self.approx_state_exp_sampling == 0:
                    pb = self.batch_beliefs[:, None, :, None] * self.pomdp.OModel.observation_tensor[None, :, :, :]  # shape N, O, S, A
                    pb_sum = torch.sum(pb, dim=2)  # shape N, O, A
                    evf_j = torch.sum(vf_j * pb_sum, dim=1)  # shape N, A
                else:
                    raise AttributeError("Batch rcs are of shape dim 4 but should be approximated...")
                advals += -self.mu * (evf_j - vf) / self.log_discount
            else:
                raise AttributeError("self.batch_rcs must have 3 or 4 dimensions but has {}.".format(self.batch_rcs.ndim))

        # advantage values for batch
        advals_max = advals.max(dim=1).values
        loss = advals_max @ advals_max

        return loss, advals_max, advals


class WonhamCollocationHJBModel(CollocationHJBModel, WonhamDiffusion):

    def __init__(self, pomdp: POMDP, batch_size: int, value_net: ValueNet, belief_prior: torch.Tensor = None,
                 approx_state_exp_sampling=1, **kwargs) -> None:
        super().__init__(pomdp=pomdp, batch_size=batch_size, value_net=value_net, belief_prior=belief_prior,
                         approx_state_exp_sampling=approx_state_exp_sampling, TModel=pomdp.TModel, OModel=pomdp.OModel,
                         **kwargs)

        # parameters to store sampled batch and precomputed quantities for batch
        self.batch_mu = None
        self.batch_sigma_outer = None

        drift_ext = self.pomdp.OModel.drift_tensor.permute(2, 1, 0)[None, :, :, :]
        dispersion_matrix = self.pomdp.OModel.dispersion_matrix
        transition_matrix = self.pomdp.TModel.transitionMatrix

        self.register_buffer('drift_ext', drift_ext)
        self.register_buffer('dispersion_matrix', dispersion_matrix)
        self.register_buffer('transition_matrix', transition_matrix)

    def resample(self):
        super().resample()

        assert (isinstance(self.pomdp.OModel, ContinuousObservationModel))
        assert (self.pomdp.OModel.drift_tensor.ndim == 3)
        assert (isinstance(self.batch_beliefs, torch.Tensor))
        assert (isinstance(self.drift_ext, torch.Tensor))

        with torch.no_grad():
            # mu_vec is of shape [N, A, S], sigma of shape [N, A, S, Noise]
            self.batch_mu, sigma = self.diffusion_dynamics(self.batch_beliefs, compute_drift=True,
                                                           compute_dispersion=True,
                                                           approx_state_exp_sampling=self.approx_state_exp_sampling)

            # self.sigma_outer = torch.sum(self.sigma[:, :, :, None, :] * self.sigma[:, :, None, :, :], dim=-1)
            self.batch_sigma_outer = sigma @ sigma.permute(0, 1, 3, 2)  # of shape N A S S

    def forward(self):
        """
        Computes the advantage values and their loss of the value network
        :return: The loss, advantage values, advantage values for all actions
        """
        vf, vg, vh = self.value_net(self.batch_beliefs, compute_hessian=True)
        hterm = torch.sum(vh[:, None, :, :] * self.batch_sigma_outer, dim=[2, 3]) / 2
        gterm = torch.sum(self.batch_mu * vg[:, None, :], dim=-1)

        # advantage values for all actions
        advals_a = self.batch_rewards - vf - gterm/self.log_discount - hterm//self.log_discount

        # advantage values for batch
        advals = advals_a.max(dim=1).values
        loss = advals @ advals

        return loss, advals, advals_a


class ExactAdvantageFunction:

    def __init__(self, pomdp, value_net):
        self.pomdp = pomdp
        self.value_net = value_net
        assert (isinstance(pomdp.OModel, RandomFiniteObservationModel))

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        batch_size = inputs.shape[0]

        def belief_sample_fn(num):
            assert (num == batch_size)
            return inputs

        hjb = DiscreteObservationCollocationHJBModel(self.pomdp, batch_size, self.value_net,
                                                     approx_state_exp_sampling=0,
                                                     belief_sample_fn=belief_sample_fn)

        hjb.resample()
        _, _, advals = hjb.forward()
        ad = advals.detach()
        advantage = ad - ad.max(dim=-1, keepdim=True)[0]
        return advantage
