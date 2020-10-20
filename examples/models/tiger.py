from packages.types.spaces import FiniteIntSpace
from packages.models.components.transitions import TabularTransitionModel
from packages.models.components.observations import RandomFiniteObservationModel
from packages.models.components.rewards import TabularRewardModel
from packages.models.pomdps.models import POMDP
import torch


class TigerProblem(POMDP):
    """ This class represents the Tiger Problem as introduced in Cassandra, A. R., Kaelbling, L. P., & Littman,
    M. L. (1994). Acting optimally in partially observable stochastic domains. In AAAI."""

    def __init__(self, discount=.9, rate_observation=2., rate_state_change=0., omodel_type='discrete'):
        """
        Initializes the POMDP
        :param discount: the discount factor between 0 and 1
        :param rate_observation: the observation rate
        :param rate_state_change: The rate for state changes should usually chosen extremely small so that the state
                                  almost never changes
        :param omodel_type: 'discrete' or 'continuous' (discrete observation model or continuous observation model)
        """
        reward_open_correct = 10
        reward_open_wrong = -100
        reward_listening = -1

        self.rate_observation = rate_observation
        self.rate_state_change = rate_state_change
        self.omodel_type = omodel_type
        self.a_labels = ['Listen', 'Open left', 'Open right']

        p_listening_correct = .85

        # initialize objects
        # state space
        n_states = 2
        s_space = FiniteIntSpace(n_states)

        # action space
        n_actions = 3
        a_space = FiniteIntSpace(n_actions)

        # transition model
        transition_matrix = torch.zeros((n_states, n_states, n_actions))
        transition_matrix[1, 0, :] = rate_state_change
        transition_matrix[0, 1, :] = rate_state_change
        transition_matrix[torch.arange(n_states), torch.arange(n_states), :] = -torch.sum(transition_matrix[:, :, :], dim=0)
        t_model = TabularTransitionModel(s_space, a_space, transition_matrix)

        n_observations = 2
        o_space = FiniteIntSpace(n_observations)

        observation_tensor = torch.ones(n_observations, n_states, n_actions)

        observation_tensor[0, 0, 0] = p_listening_correct
        observation_tensor[1, 0, 0] = 1 - p_listening_correct
        observation_tensor[0, 1, 0] = 1 - p_listening_correct
        observation_tensor[1, 1, 0] = p_listening_correct
        observation_tensor /= torch.sum(observation_tensor, axis=0, keepdim=True)
        o_model = RandomFiniteObservationModel(s_space, a_space, o_space, rate_observation, observation_tensor)

        # Reward Model
        # the goal is to be in state 1
        reward_matrix = torch.ones((n_states, n_actions)) * reward_listening
        reward_matrix[0, 1] = reward_open_correct
        reward_matrix[0, 2] = reward_open_wrong
        reward_matrix[1, 1] = reward_open_wrong
        reward_matrix[1, 2] = reward_open_correct

        r_model = TabularRewardModel(s_space, a_space, reward_matrix)

        # POMDP model
        super().__init__(s_space, a_space, o_space, t_model, o_model, r_model, discount)
