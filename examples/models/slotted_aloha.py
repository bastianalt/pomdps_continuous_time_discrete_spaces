from packages.models.pomdps.models import POMDP
from packages.types.spaces import FiniteIntSpace
from packages.models.components.transitions import TabularTransitionModel
from packages.models.components.observations import RandomFiniteObservationModel
from packages.models.components.rewards import TabularRewardModel
import torch.distributions
import torch


class SlottedAloha(POMDP):

    def __init__(self, n_pack=10, n_tstates=3, pack_arrival_rate=.5, send_rate=5., observation_rate=.5, discount=.9):

        self.n_transition_states = n_tstates
        self.n_packages = n_pack
        self.pack_arrival_rate = pack_arrival_rate
        self.send_rate = send_rate
        self.observation_rate = observation_rate

        n_a = n_pack - 1
        self.n_actions = n_a

        # range tensors
        a_all = torch.arange(n_a)
        p_all = torch.arange(n_pack)
        ts_all = torch.arange(n_tstates)

        a = 1/(a_all + 1).float()
        self.actions = a

        # generate transition model
        t = torch.zeros(n_tstates, n_pack, n_tstates, n_pack, n_a)
        for p in range(n_pack):
            for ai in range(n_a):
                pa = a[ai]

                # idle state
                t[TransitionStates.IDLE, p, :, p, ai] = send_rate * (1 - pa) ** p

                # successful transmission
                if p > 0:
                    t[TransitionStates.TRANSMIT, p - 1, :, p, ai] = send_rate * pa * (1-pa) ** (p-1)

                # collision
                if p > 1:
                    t[TransitionStates.COLLISION, p, :, p, ai] = send_rate * (1 - pa * (1 - pa) ** (p - 1) - (1 - pa) ** p)

                # new package arrives
                if p < n_pack - 1:
                    t[ts_all, p+1, ts_all, p, ai] = pack_arrival_rate

                for st in range(n_tstates):
                    t[st, p, st, p, ai] -= torch.sum(t[:, :, st, p, ai], dim=[0, 1])
                    torch.allclose(torch.sum(t[:, :, st, p, ai], dim=[0, 1]), torch.as_tensor(0.), atol=1e-06)

                assert(torch.allclose(torch.sum(t[:, :, :, p, ai], dim=[0, 1]), torch.as_tensor(0.), atol=1e-06))

        # initialize objects
        # state space
        self.n_states = n_pack * n_tstates
        s_space = FiniteIntSpace(self.n_states)

        # action space
        self.n_actions = n_a
        a_space = FiniteIntSpace(self.n_actions)

        # transition model
        transition_matrix = t.reshape(self.n_states, self.n_states, n_a)
        t_model = TabularTransitionModel(s_space, a_space, transition_matrix)

        # observation model
        n_observations = n_tstates
        o_space = FiniteIntSpace(n_observations)
        observation_tensor = torch.zeros(n_tstates, n_tstates, n_pack, n_a)
        observation_tensor[ts_all, ts_all] = 1
        observation_tensor_flatstates = observation_tensor.reshape(n_tstates, n_tstates * n_pack, n_a)
        o_model = RandomFiniteObservationModel(s_space, a_space, o_space, observation_rate, observation_tensor_flatstates)

        # Reward Model
        # the goal is to be in state 0
        reward_mult = 1.
        reward_matrix = reward_mult * (n_pack - p_all.float() - 1)[None, :, None].repeat(n_tstates, 1, n_a)
        reward_matrix_flatstates = reward_matrix.reshape(-1, n_a)
        r_model = TabularRewardModel(s_space, a_space, reward_matrix_flatstates)

        # POMDP model
        super().__init__(s_space, a_space, o_space, t_model, o_model, r_model, discount)

    def get_parameters(self):
        params = super().get_parameters()
        params['n_tstates'] = self.n_transition_states
        params['n_pack'] = self.n_packages
        params['pack_arrival_rate'] = self.pack_arrival_rate
        params['observation_rate'] = self.observation_rate
        params['send_rate'] = self.send_rate
        return params


class TransitionStates:
    IDLE = 0
    TRANSMIT = 1
    COLLISION = 2
