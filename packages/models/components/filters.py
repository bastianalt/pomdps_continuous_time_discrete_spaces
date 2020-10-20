from abc import ABC, abstractmethod
from packages.types.spaces import FiniteIntSpace
from packages.models.components.transitions import TabularTransitionModel
from packages.models.components.observations import RandomDiscreteDensityObservationModel, \
    DiscreteObservationProcess, ContinuousObservationModel, ContinuousObservationProcess
from packages.models.pomdps.policys import DeterministicPolicy
from packages.utils.utils import concatenateODESolution
import torch
from torchdiffeq import odeint
from torch.distributions import Poisson
# noinspection PyProtectedMember
from torchdiffeq._impl.solvers import _FixedGridState, _solution2interpolants, OdeTorchSolution
import copy
from packages.models.components.dynamics import WonhamDiffusion, PiecewiseDeterministicMarkov
from typing import Dict
from collections import namedtuple

Belief=namedtuple('Belief', ('time', 'belief', 'belief_plus'))
Observation = namedtuple('Observation', ('time', 'observation'))

class Filter(ABC):
    def __init__(self, SSpace: FiniteIntSpace, **kwargs):
        """
        Abstract base class for a filter
        :param SSpace: State Space of the filter
        """

        super().__init__(**kwargs)
        self.belief_shape = SSpace.cardinalities
        self.SSpace = SSpace

    # TODO update
    @abstractmethod
    def sample(self, initial_belief: torch.Tensor, initial_observation: torch.Tensor, t_grid: torch.Tensor,
               fixed_state: torch.Tensor, ode_options: Dict):
        """
        Method which evolves filter according to prior and samples random observations
        :param ode_options:
        :param initial_belief: belief
        :param initial_observation: observation at t_start (only important for continuous observation process)
        :param t_span: [t_start, t_end] start time of the filter and end time of the filter, respectively
        :param fixed_state: state, which is used for conditioning to draw random observations
        :return:
        """
        pass


class ContinuousRandomDiscreteFilter(Filter, PiecewiseDeterministicMarkov):
    def __init__(self, SSpace: FiniteIntSpace,
                 ASpace: FiniteIntSpace,
                 TModel: TabularTransitionModel,
                 OModel: RandomDiscreteDensityObservationModel,
                 pi: DeterministicPolicy):
        """
        Filter, which performs exact filtering for the belief
        :param SSpace: State space of the filter
        :param ASpace: Action space of the filter
        :param TModel: underlying transition model used for the prior dynamic
        :param OModel: observation model, used for filtering and generation of observations
        :param pi: policy is used for conditioning the transition model on actions
        """

        assert TModel.SSpace == SSpace
        assert TModel.ASpace == ASpace
        assert OModel.SSpace == SSpace
        assert OModel.ASpace == ASpace
        assert pi.SSpace == SSpace
        assert pi.ASpace == ASpace

        super().__init__(SSpace=SSpace, TModel=TModel, OModel=OModel)

        self.ASpace = ASpace
        self.pi = pi

    def _prior_dynamic(self, t: torch.Tensor, belief: torch.Tensor) -> torch.Tensor:
        """
              Kolmogorov ode for prior dynamic
        """
        return self.TModel.transitionMatrix[..., self.pi(belief, t=t)] @ belief

    # TODO update doc
    def sample(self,
               initial_belief: torch.Tensor,
               initial_observation: torch.Tensor,
               t_grid: torch.Tensor,
               fixed_state: torch.Tensor,
               ode_options: Dict = None):
        """
             Method which evolves filter according to prior and samples random observations
             :param ode_options:
             :param initial_belief: belief
             :param initial_observation: not used in ContinuousRandomDiscreteFilter
             :param t_span: [t_start, t_end] start time of the filter and end time of the filter, respectively
             :param fixed_state: state, which is used for conditioning to draw random observations
             :return: belief: callable function which returns belief state for a given time b=belief(t)
                      observations: observations sampled during t_span
             """
        if ode_options is None:
            ode_options = dict()

        device = initial_belief.device

        t = t_grid[0]
        T = t_grid[-1]
        belief_t = initial_belief

        # Sample observations
        rv = Poisson(self.OModel.observationRate * (T - t))
        observationTimes = torch.unique(
                torch.rand(int(rv.sample()), device=device) * (T - t) + t)  # Remove observation times at the same value

        observation_traj=[]
        belief_traj=[]

        for i, t_i in enumerate(observationTimes):
            t_end = t_i
            t_mask = (t < t_grid) * (t_grid < t_end)
            t_grid_i = torch.cat((t.view(-1), t_grid[t_mask].view(-1), t_end.view(-1)))
            ode_result = odeint(self._prior_dynamic, belief_t, t_grid_i, **ode_options)

            belief_t_minus = ode_result[-1, :]
            # Renormalization for numerical reasons
            belief_t_minus[belief_t_minus < 0] = 0
            belief_t_minus /= torch.sum(belief_t_minus)

            observation = self.OModel.draw(fixed_state, self.pi(belief_t_minus, t=t_end))

            log_belief_t = self.OModel(observation, torch.arange(self.SSpace.nElements, device=device),
                                       self.pi(belief_t_minus, t=t_end)) + torch.log(belief_t_minus)
            log_Z_t = torch.logsumexp(log_belief_t, 0)
            belief_t = torch.exp(log_belief_t - log_Z_t)

            for j,t_j in enumerate(t_grid_i[:-1]):
                belief_traj.append(Belief(t_j.clone(), ode_result[j, :].clone(),ode_result[j, :].clone()))
            belief_traj.append(Belief(t_i.clone(), belief_t_minus.clone(),belief_t.clone()))

            observation_traj.append(Observation(t_i.clone(), observation.clone()))
            t = t_i

        t_mask = (t < t_grid) * (t_grid < T)
        t_grid_i=torch.cat((t.view(-1), t_grid[t_mask].view(-1), T.view(-1)))
        ode_result = odeint(self._prior_dynamic, belief_t, t_grid_i ,
                            **ode_options)
        for j, t_j in enumerate(t_grid_i):
            belief_traj.append(Belief(t_j.clone(), ode_result[j, :].clone(),ode_result[j, :].clone()))

        return belief_traj, observation_traj


class WonhamFilter(Filter, WonhamDiffusion):
    def __init__(self, SSpace: FiniteIntSpace,
                 ASpace: FiniteIntSpace,
                 TModel: TabularTransitionModel,
                 OModel: ContinuousObservationModel,
                 pi: DeterministicPolicy):
        """
        Wonham-filter which performs exact filtering for the belief
        :param SSpace: State space of the filter
        :param ASpace: Action space of the filter
        :param TModel: underlying transition model used for the prior dynamic
        :param OModel: observation model, used for filtering and generation of observations
        :param pi: policy is used for conditioning the transition model on actions
        """

        assert TModel.SSpace == SSpace
        assert TModel.ASpace == ASpace
        assert OModel.SSpace == SSpace
        assert OModel.ASpace == ASpace
        assert pi.SSpace == SSpace
        assert pi.ASpace == ASpace

        super().__init__(SSpace=SSpace, TModel=TModel, OModel=OModel)

        self.ASpace = ASpace
        self.pi = pi

    def _prior_dynamic_inc(self, belief: torch.Tensor, action, dt) -> torch.Tensor:
        return self.TModel.transitionMatrix[:, :, action] @ belief * dt

    def _expected_drift_vector(self, belief, action):
        return self.OModel.drift_tensor[:, :, action] @ belief

    def _belief_inc(self, do, belief, action, dt):
        b_prior_inc = self._prior_dynamic_inc(belief, action, dt)

        # Kushner-Stratonovic
        g_bar = self._expected_drift_vector(belief, action)
        belief_inc = b_prior_inc + belief * (
                (self.OModel.drift_tensor[:, :, action] - g_bar).T @ self.outer_dispersion_inv @ (
                do - g_bar * dt)).squeeze()
        return belief_inc

    def _belief_inc_unnorm(self, do, belief, action, dt):
        b_prior_inc = self._prior_dynamic_inc(belief, action, dt)

        # zakkai
        belief_inc_unnorm = b_prior_inc + belief * (
                self.OModel.drift_tensor[:, :, action].T @ self.outer_dispersion_inv @ do).squeeze()
        return belief_inc_unnorm

    def _sample_observation_inc(self, state, action, dt):
        dw = torch.randn(self.OModel.noise_dim, device=dt.device) * torch.sqrt(dt)
        return self.OModel.drift_tensor[:, state, action] * dt + self.OModel.dispersion_matrix @ dw

    #TODO fix this!
    def sample(self,
               initial_belief: torch.Tensor,
               initial_observation: torch.Tensor,
               t_grid: torch.Tensor,
               fixed_state: torch.Tensor, ode_options: Dict = None):
        """
        Method which evolves filter and samples a random observation process using Euler-Maruyama

        :param initial_belief: belief used as initialization of the filter
        :param initial_observation: Initial observation used for the observation process if None zero is used as
                                    initialization
        :param t_span: [t_start, t_end] start time of the filter and end time of the filter, respectively
        :param fixed_state: state, which is used for conditioning to draw random observations
        :param ode_options: ode_options['dt'] time increment for Euler-Maruyama
        :return:  belief: callable function which returns belief state for a given time b=belief(t)
                  observations: observations sampled during t_span
        """

        if initial_observation is None:
            initial_observation = torch.zeros(self.OModel.OSpace.dimension)

        # Select the step size as the 10*max_rate
        if ode_options is None:
            ode_options = dict()

        try:
            dt = ode_options['dt']
        except KeyError:
            dt = torch.min(1 / (self.TModel.max_rate(fixed_state) * 10),torch.tensor(1e-10,device=self.TModel.max_rate(fixed_state).device))

        t=t_grid[0]
        T = t_grid[-1]
        dt = torch.as_tensor(dt)

        observation_traj=[]
        belief_traj=[]

        observation_t = initial_observation
        belief_t = initial_belief

        while t <= T:
            belief_traj.append(Belief(t.clone(),belief_t.clone()))
            observation_traj.append(Observation(t.clone(), observation_t.clone()))

            action = self.pi(belief_t, t=t)
            do = self._sample_observation_inc(fixed_state, action, dt)
            db = self._belief_inc(do, belief_t, action, dt)

            belief_new = belief_t + db
            belief_new[belief_new < 0] = 0
            belief_new /= belief_new.sum()


            observation_new = observation_t + do.squeeze()

            belief_t = belief_new
            observation_t = observation_new
            t += dt

        return belief_traj,observation_traj, []
