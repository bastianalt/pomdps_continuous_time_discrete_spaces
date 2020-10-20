from typing import Union, Optional, Dict, List, Tuple
import torch
from packages.models.pomdps.models import POMDP
from packages.models.pomdps.policys import Policy
from packages.models.components.filters import Filter
from packages.types.spaces import FiniteIntSpace
from collections import namedtuple

ObservedData = namedtuple('ObservedData', ('t', 'belief','belief_plus', 'belief_derivative', 'action', 'reward'))
State = namedtuple('State', ('t_start', 't_end', 'state'))


class POMDPSimulator:
    def __init__(self, pomdp: POMDP):
        self.pomdp = pomdp

    # TODO Update Doc
    def sampleTraj(self, t_grid: torch.Tensor,
                   pi: Policy,
                   filter: Filter,
                   initial_belief: torch.Tensor,
                   start_state: Optional[Union[torch.Tensor, int]] = None,
                   ode_options: Dict = None) -> Tuple[List[State], List[ObservedData]]:
        """
        Creates a sample trajectory for the POMDP for the grid t_grid
        :param ode_options:
        :param t_grid: torch tensor representing a grid of time points used for simulation
        :param pi: policy
        :param filter: filter used by the policy
        :param initial_belief: starting belief for the filter
        :param start_state: start state of the markov process for simulation
        :return: Traj: object containing
                states: List of states visited by the markov process
                jump_times: jump times of the markov process
                belief: callback function which returns the belief state for each time point
                actions calbback function which returns the selected action for each time point
                observations: observations sampled during the time interval [0,T]
                rewards: Callback function which returns the reward rate for a each time point
        """
        if ode_options is None:
            ode_options = dict()
        # Get device
        device = initial_belief.device

        # Todo assert t_grid increasing
        t_grid = torch.as_tensor(t_grid, device=device)
        # End time
        T = t_grid[-1]

        #TODO Include initial distribution to POMDP
        if start_state is None:
            if isinstance(self.pomdp.SSpace, FiniteIntSpace):  # Start with random state if no start state is given
                start_state = torch.randint(self.pomdp.SSpace.nElements, (1,), device=device)
            else:
                start_state = torch.tensor([0], device=device)
            # Start with zero state if no start state is given
        else:
            start_state = torch.as_tensor(start_state, device=device).view(-1)

        # Start of the simulation
        t = t_grid[0]
        old_state = start_state
        initial_observation = None  # last observation --> only relevant for continuous observation process

        observed_traj = []
        latent_traj = []
        observation_traj = []

        while t <= T:
            # Do a simulation for every jump of the markov process
            new_state, new_waiting_time, new_observed_traj, new_observation_traj = self._simulateTransition(
                pi, filter, initial_belief,
                initial_observation, old_state,
                t, t_grid, ode_options)

            # Save the trajectory
            latent_traj.append(State(t.clone(), (t + new_waiting_time).clone(), old_state.clone()))
            observed_traj += new_observed_traj[:-1]
            observation_traj += new_observation_traj

            t += new_waiting_time
            if t >= T:
                observed_traj.append(new_observed_traj[-1])
                break

            old_state = new_state
            initial_belief = new_observed_traj[-1].belief
            if new_observation_traj:
                initial_observation = new_observation_traj[-1]
            else:
                initial_observation = None

        return latent_traj, observed_traj,observation_traj

    def _simulateTransition(self, pi: Policy,
                            filter: Filter,
                            initial_belief: torch.Tensor,
                            initial_observation: torch.Tensor,
                            old_state: torch.Tensor,
                            t_start: torch.Tensor,
                            t_grid: torch.Tensor,
                            ode_options: Dict = None) -> List:
        """
        Simulates the markov process from t_start to the next jump
        :param pi: policy used for simulation
        :param filter: filter used for simulation
        :param initial_belief: belief state at t_start
        :param initial_observation: observation at t_start (only important for continuous observation process)
        :param old_state: state at which the markov process starts
        :param t_start: start time of the markov process (last jump time)
        :return: new_state: Next state of the markov process
                waiting_time: waiting time bnetween jumps
                belief: callback function which returns for a time "t" in the time interval [t_start, t_start + waiting_time] the belief b(t)
                observationsTransition:  observations sampled in the interval [t_start, t_start + waiting_time]
        """
        if ode_options is None:
            ode_options = dict()
        # Get device
        device = initial_belief.device

        rate_max = self.pomdp.TModel.max_rate(old_state)
        t = t_start.clone()
        T = t_grid[-1]
        observed_traj = []
        observation_traj = []

        # Simulate inhomogenous markov process using thinning
        while True:
            # Sample random number
            u = torch.rand(1, device=device).squeeze(dim=0)
            if rate_max == 0:
                waiting_time_min_thinned = torch.tensor(float("Inf"), device=device)
            else:
                waiting_time_min_thinned = -torch.log(u) / rate_max
            waiting_time_max_thinned = T - t
            waiting_time_thinned = torch.min(waiting_time_min_thinned, waiting_time_max_thinned)
            t_mask = (t < t_grid) * (t_grid < t + waiting_time_thinned)
            new_belief_traj, new_observation_traj = filter.sample(initial_belief, initial_observation,
                                                                           torch.cat((t.view(-1),
                                                                                      t_grid[t_mask].view(-1),
                                                                                      (t + waiting_time_thinned).view(
                                                                                          -1))), old_state,
                                                                           ode_options)
            observation_traj += new_observation_traj
            for i, belief_obj in enumerate(new_belief_traj):
                belief = belief_obj.belief
                belief_plus = belief_obj.belief_plus
                t_i = belief_obj.time
                action = pi(belief, t=t_i)
                reward = self.pomdp.RModel(old_state, action)
                belief_derivative = torch.sum(self.pomdp.TModel.transitionMatrix[:, :, action] * belief[None, :], dim=1)

                observed_traj.append(ObservedData(t_i.clone(), belief.clone(),belief_plus.clone(), belief_derivative.clone(),
                                                  action.clone(), reward.clone()))

            t += waiting_time_thinned
            if torch.rand(1, device=device) <= -self.pomdp.TModel(old_state, old_state,
                                                                  pi(new_belief_traj[-1].belief, t=t)) / rate_max or t == T:
                break
            initial_belief = new_belief_traj[-1].belief
            if new_observation_traj:
                initial_observation = new_observation_traj[-1].observation

        waiting_time = t - t_start
        new_state = self.pomdp.TModel.draw(old_state, pi(new_belief_traj[-1].belief, t=t))

        return new_state, waiting_time, observed_traj, observation_traj
