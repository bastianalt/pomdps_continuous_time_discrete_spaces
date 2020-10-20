from abc import ABC, abstractmethod
from packages.types.spaces import Space, FiniteIntSpace
import torch
import torch.nn as nn


# Classes for transition models used in pomdps
class TransitionModel(ABC, nn.Module):
    def __init__(self, SSpace: Space, ASpace: Space):
        """
        Abstract base class for a transition model
        :param SSpace: State space used in the Transition Model
        :param ASpace: Action space used in the Transition Model
        """
        self.SSpace = SSpace
        self.ASpace = ASpace
        nn.Module.__init__(self)

    @abstractmethod
    def __call__(self, newState: torch.Tensor, oldState: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        return transition rate value for newState given oldState and action, i.e. T(s',s,a)
        """
        pass

    @abstractmethod
    def max_rate(self, state: torch.Tensor) -> torch.Tensor:
        """
        For a given state return the maximum transition rate over all actions
        """
        pass

    @abstractmethod
    def draw(self, oldState: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Draw a new state conditioned on old state and action s' ~ p(s'|s,a)
        """
        pass


class TabularTransitionModel(TransitionModel):
    def __init__(self, SSpace: FiniteIntSpace, ASpace: FiniteIntSpace, transitionMatrix: torch.Tensor):
        """
        Tabular model for a transition model with multidimensional state and action spaces
        :param SSpace: State space used in the Transition Model
        :param ASpace: Action space used in the Transition Model
        :param transitionMatrix:
            (SSpace_1 x SSpace_2 x... SSpace_n x SSpace_1 x SSpace_2 x... SSpace_n x ASpace_1 x ASpace_2 x... ASpace_m)
            tabular model of the transition matrix (nextState, oldState, action)
        """
        super().__init__(SSpace, ASpace)
        self.transitionWeights = None
        self.transitionMatrix = transitionMatrix

    @property
    def transitionMatrix(self):
        return self._transitionMatrix

    # noinspection SpellCheckingInspection
    @transitionMatrix.setter
    def transitionMatrix(self, x: torch.Tensor):
        assert isinstance(self.SSpace, FiniteIntSpace)
        assert isinstance(self.ASpace, FiniteIntSpace)
        assert x.shape == torch.Size([self.SSpace.nElements, self.SSpace.nElements, self.ASpace.nElements])
        self.register_buffer('_transitionMatrix', x)

        # Set Weight matrix for the underlying categorical distribution

        # TODO sparse weight matrix is still very  slow
        # weights = None
        # if isinstance(x, sparse.COO):
        #     data=[]
        #     coordsT=[]
        #     for i, coord in enumerate(x.coords.T):
        #         new_state=(int(coord[slice(self.SSpace.dimension)]),)
        #         old_state=(int(coord[slice(self.SSpace.dimension,2*self.SSpace.dimension)]),)
        #         action=(int(coord[slice(2*self.SSpace.dimension,2*self.SSpace.dimension+self.ASpace.dimension)]),)
        #         if old_state!=new_state:
        #             coordsT.append(coord)
        #             data.append(-x.data[i]/x[old_state+old_state+action])
        #     weights=sparse.COO(np.asarray(coordsT).T,np.asarray(data),shape=x.shape)
        #
        # else:
        #     weights = np.zeros_like(x)
        #     for action in np.ndindex(self.ASpace.cardinalities):
        #         for old_state in np.ndindex(self.SSpace.cardinalities):
        #             rate = -x[old_state + old_state + action]
        #             for next_state in np.ndindex(self.SSpace.cardinalities):
        #                 if next_state != old_state:
        #                     weights[next_state + old_state + action] = x[next_state + old_state + action] / rate

        transitionWeights = torch.zeros_like(x)
        for action in range(self.ASpace.nElements):
            transitionWeights[:, :, action] = -x[:, :, action] / x[:, :, action].diag()
            transitionWeights[:, :, action] += torch.eye(self.SSpace.nElements).to(x)
        self.transitionWeights = transitionWeights

    @property
    def transitionWeights(self):
        return self._transitionWeights

    @transitionWeights.setter
    def transitionWeights(self, x: torch.Tensor):
        if x is None:
            # noinspection PyTypeChecker
            self.register_buffer('_transitionWeights', x)
            return

        # assert torch.all(x >= 0)
        # assert torch.allclose(torch.sum(x, dim=list(range(self.SSpace.dimension))), torch.as_tensor(1.0))
        self.register_buffer('_transitionWeights', x)

    def __call__(self, newState: torch.Tensor, oldState: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.transitionMatrix[newState, oldState, action]

    def max_rate(self, state: torch.Tensor) -> torch.Tensor:
        return torch.max(-self.transitionMatrix[state, state, :])

    def draw(self, oldState: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        p=self.transitionWeights[:, oldState, action].squeeze()
        if not torch.any(torch.isnan(p)):
            new_state=torch.multinomial(p, 1)
        else:
            new_state=oldState

        return new_state
