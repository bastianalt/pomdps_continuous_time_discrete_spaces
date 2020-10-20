from typing import Union, Optional, Any
from abc import ABC, abstractmethod

from packages.models.pomdps.learners.collocation import ExactAdvantageFunction
from packages.types.spaces import Space, FiniteIntSpace
from packages.models.components.advantages import AdvantageNet
import copy
import torch
import torch.nn as nn
from torchdiffeq._impl.solvers import _FixedGridState, _solution2interpolants, OdeTorchSolution

class Policy(ABC, nn.Module):
    def __init__(self, SSpace: Space, ASpace: Space):
        """
        Abstract base class for a policy object
        :param SSpace: The state space of the policy
        :param ASpace:  The action space of the policy
        """
        self.SSpace = SSpace
        self.ASpace = ASpace
        super().__init__()

    @abstractmethod
    def __call__(self, belief: torch.Tensor, t: Optional[torch.Tensor] = None):
        """
        For a given belief returns an action at time t
        """
        pass


class DeterministicPolicy(Policy, ABC):
    def __init__(self, SSpace: FiniteIntSpace, ASpace: FiniteIntSpace):
        """
        Policy class for determinsitic policies return action a=pi(belief)
        :param SSpace: The state space of the policy
        :param ASpace: The action space of the policy
        :param policy_function: function which returns an action for a belief state
        """
        super().__init__(SSpace, ASpace)


class AdvantagePolicy(DeterministicPolicy):
    def __init__(self, SSpace: FiniteIntSpace, ASpace: FiniteIntSpace, advantage_net: AdvantageNet = None):
        """
        Policy parametrized by a NN Advantage function. Selects the argmax of the advantage function as action.
        Pi(s)=argmax_a A(s,a)
        :param SSpace: State space of the policy needed for representing the belief
        :param ASpace: Action space of the policy needed for representing the belief
        :param advantage_net: NN representing the advantage function
        """
        super(AdvantagePolicy, self).__init__(SSpace, ASpace)

        if advantage_net is None:
            advantage_net = AdvantageNet(SSpace.nElements, ASpace.nElements)

        self.advantage_net = advantage_net
        self.exploration = False
        self.perturbation = None

    def greedy(self):
        """
        Sets the policy to greedy mode (no exploration)
        """
        self.exploration = False
        self.perturbation = None

    def ou_perturbation(self, T: torch.Tensor, dt: Optional[Union[Any, torch.Tensor]] = 1e-2,
                        kappa: Optional[Union[Any, torch.Tensor]] = None,
                        sigma: Optional[Union[Any, torch.Tensor]] = None):
        """
        Adds an Ornstein-Uhlenbeck perturbation trajectory in the interval [0,T] to the policy as
        dz=-kappa*x dt + sigma dw and pi_t(s)=argmax(A(a,s)+z_t(a))

        :param T: End point of the perturbation
        :param dt: time increment used for Euler-Maryuama
        :param kappa: decay factor of the Ornstein-Uhlenbeck process
        :param sigma: dispersion  of the Ornstein-Uhlenbeck process
        """
        if kappa is None:
            kappa = 7.5
        if sigma is None:
            sigma = 1.5

        # Draw from an ornstein ulenbeck prcoess
        device = T.device
        dt = torch.as_tensor(dt, device=device, dtype=torch.float)
        kappa = torch.as_tensor(kappa, device=device, dtype=torch.float)
        sigma = torch.as_tensor(sigma, device=device, dtype=torch.float)

        intp_states = []
        t = torch.zeros(1, device=device)
        # Draw initial from stationary distribuion of OU-process for initial pertubation
        assert isinstance(self.ASpace, FiniteIntSpace)
        perturbation_t = sigma * torch.randn(self.ASpace.nElements, device=device) * torch.sqrt(2 * kappa)
        while t <= T:
            perturbation_inc = -kappa * perturbation_t * dt + sigma * torch.randn_like(perturbation_t) * torch.sqrt(dt)
            intp_states.append(
                    copy.deepcopy(_FixedGridState(t, t + dt, (perturbation_t,), (perturbation_t + perturbation_inc,))))
            perturbation_t += perturbation_inc
            t += dt

        ts, interpolants = _solution2interpolants([intp_states])
        self.perturbation = OdeTorchSolution(ts.squeeze(dim=1), interpolants, interpolator='_linear_interp')
        self.exploration = True

    def __call__(self, belief: torch.Tensor, t: Optional[torch.Tensor] = None):
        if self.exploration and t is not None:
            return (self.advantage_net(belief) + self.perturbation(t)).argmax()
        else:
            return self.advantage_net(belief).argmax()


class AdvantageValuePolicy(DeterministicPolicy):
    def __init__(self, pomdp, value_net):
        super(AdvantageValuePolicy, self).__init__(pomdp.SSpace, pomdp.ASpace)

        self.pomdp = pomdp
        self.value_net = value_net
        self.advantage_function = ExactAdvantageFunction(pomdp, value_net)

    def __call__(self, belief: torch.Tensor, t: Optional[torch.Tensor] = None):
        if belief.ndim == 1:
            belief = belief.reshape(1, -1)
        advals = self.advantage_function(belief)
        return advals.argmax()
