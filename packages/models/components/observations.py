from typing import Union, List
from abc import ABC, abstractmethod
import torch
from torch.distributions import Categorical
from packages.types.spaces import Space, FiniteIntSpace
from packages.utils.utils import concatenateODESolution
# noinspection PyProtectedMember
from torchdiffeq._impl.solvers import OdeTorchSolution
import torch.nn as nn
from torch.distributions import Normal


# Classes for observation models used in POMDPs

class ObservationModel(ABC, nn.Module):
    def __init__(self, SSpace: Space, ASpace: Space, OSpace: Space):
        """
        Abstract base class for Observation Models
        :param SSpace: State space used in the Observation Model
        :param ASpace: Action space used in the Observation Model
        :param OSpace: Observation space used in the Observation Model
        """
        self.SSpace = SSpace
        self.ASpace = ASpace
        self.OSpace = OSpace
        nn.Module.__init__(self)
    #
    # @abstractmethod
    # def __call__(self, observation: torch.Tensor, state: torch.Tensor,
    #              action: torch.Tensor) -> torch.Tensor:
    #     """
    #     return log density or log probability value for observation log p(o|s,a)
    #     """
    #     pass
    #
    # @abstractmethod
    # def draw(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    #     """
    #     Draw an observation condition on state and action o ~ p(o|s,a)
    #     """
    #     pass


class RandomDiscreteDensityObservationModel(ObservationModel, ABC):
    def __init__(self, SSpace: Space,
                 ASpace: Space,
                 OSpace: Space,
                 observationRate: Union[torch.Tensor, float]):
        """
        Class for an observation model defined by the log of a probability density function

        :param SSpace: State space used in the Observation Model
        :param ASpace: Action space used in the Observation Model
        :param OSpace: Observation space used in the Observation Model
        :param observationRate: rate for a poisson process defining the observation times
        """
        ObservationModel.__init__(self, SSpace, ASpace, OSpace)
        self.register_buffer('observationRate', torch.as_tensor(observationRate).double())

    @abstractmethod
    def __call__(self, observation: torch.Tensor, state: torch.Tensor,
                 action: torch.Tensor) -> torch.Tensor:
        """
        returns log density/log probability of log p(o|s,a)
        :param observation: torch tensor containing o
        :param state: torch tensor containing s
        :param action: torch tensor containing a
        :return: torch tensor of log density/ log probability log p(o|s,a)
        """
        raise NotImplementedError

    @abstractmethod
    def draw(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        returns a random draw of o ~ p(o|s,a)
        :param state: torch tensor containing s
        :param action: torch tensor containing a
        :return: torch tensor observation o
        """
        raise NotImplementedError()


class RandomDiscreteGaussianObservationModel(RandomDiscreteDensityObservationModel):

    def __init__(self, SSpace: Space,
                 ASpace: Space,
                 OSpace: Space,
                 observationRate: Union[torch.Tensor, float],
                 noise: torch.Tensor):
        """
        Discrete State space model with additive Gaussian noise on states
        :param SSpace: State space used in the Observation Model
        :param ASpace: Action space used in the Observation Model
        :param OSpace: Observation space used in the Observation Model
        :param observationRate: rate for a poisson process defining the observation times
        :param noise: Noise scale of the Gaussian noise
        """
        RandomDiscreteDensityObservationModel.__init__(self, SSpace, ASpace, OSpace, observationRate)
        self.register_buffer('noise', noise)

    def __call__(self, observation: torch.Tensor, state: torch.Tensor,
                 action: torch.Tensor) -> torch.Tensor:
        rv = Normal(loc=state.double(), scale=self.noise.double())
        return rv.log_prob(observation).float()

    def draw(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        rv = Normal(loc=state.double(), scale=self.noise.double())
        return rv.sample().float()


# TODO Comment
class RandomFiniteObservationModel(RandomDiscreteDensityObservationModel):
    def __init__(self, SSpace: FiniteIntSpace,
                 ASpace: FiniteIntSpace,
                 OSpace: FiniteIntSpace,
                 observationRate: Union[torch.Tensor, float],
                 observationTensor: torch.Tensor):
        assert torch.allclose(torch.sum(observationTensor, dim=0), torch.as_tensor(1.0))
        super().__init__(SSpace, ASpace, OSpace, observationRate)
        self.register_buffer('observation_tensor', observationTensor)

    def __call__(self, observation: torch.Tensor, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        rv = Categorical(self.observation_tensor[:, state, action].T)
        return rv.log_prob(observation).float()

    def draw(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        rv = Categorical(self.observation_tensor[:, state, action].T)
        return rv.sample().float()


class ContinuousObservationModel(ObservationModel):
    def __init__(self, SSpace: FiniteIntSpace,
                 ASpace: FiniteIntSpace,
                 OSpace: Space,
                 drift_tensor: torch.Tensor,
                 dispersion_matrix: torch.Tensor):
        """
        Continuous observation model given by the sde do=g(s,a) dt + H dw
        :param SSpace: State space used in the Observation Model
        :param ASpace: Action space used in the Observation Model
        :param OSpace: Observation space used in the Observation Model
        :param drift_tensor: drift tensor specifying g(s,a) [num_Obs x num_States x num_Actions]
        :param dispersion_matrix: dispersion matrix H [num_States x noise_dim ]
        """
        assert (OSpace.dimension, SSpace.nElements, ASpace.nElements) == tuple(drift_tensor.shape)
        assert dispersion_matrix.ndim == 2 and OSpace.dimension == dispersion_matrix.shape[0]

        super(ContinuousObservationModel, self).__init__(SSpace, ASpace, OSpace)
        self.register_buffer('drift_tensor', torch.as_tensor(drift_tensor, dtype=torch.float))
        self.register_buffer('dispersion_matrix', torch.as_tensor(dispersion_matrix, dtype=torch.float))
        self.noise_dim = dispersion_matrix.shape[1]


class ObservationProcess(ABC):
    def __init__(self):
        """base class for observation processes"""
        pass

    @abstractmethod
    def __call__(self, t):
        pass


class DiscreteObservationProcess(ObservationProcess):
    def __init__(self, observationTimes: torch.Tensor, observationValues: torch.Tensor):
        """
        Realization of a discrete observation process
        :param observationTimes: torch tensor containing the observation times of the discrete observation process
        :param observationValues:  torch tensor containing the values of the observations of the discrete observation
                                    process
        """
        super().__init__()
        self.observationTimes = observationTimes
        self.observationValues = observationValues

    def __call__(self, t):
        return None


class ContinuousObservationProcess(ObservationProcess):
    def __init__(self, observations: OdeTorchSolution):
        """
        Realization of a continuous observation process
        :param observations: OdeTorchSolution obj which returns the observation for a time point
        """
        super().__init__()
        self.observations = observations

    def __call__(self, t):
        return self.observations(t)


def concatenate_observation_process(
        observation_process_list: List[Union[DiscreteObservationProcess, ContinuousObservationProcess]]):
    """
    concatenates a list of ObservationProcesses to one coherent ObservationProcess
    :param observation_process_list: list of ObservationProcess instances
    :return: ObservationProcess instance
    """
    observation_process = None
    if observation_process_list:
        if all([isinstance(observation_process, DiscreteObservationProcess) for observation_process in
                observation_process_list]):
            observationTimes = torch.cat(
                    [observation_process.observationTimes for observation_process in observation_process_list])
            observationValues = torch.cat(
                    [observation_process.observationValues for observation_process in observation_process_list], dim=0)
            observation_process = DiscreteObservationProcess(observationTimes, observationValues)
        elif all([isinstance(observation_process, ContinuousObservationProcess) for observation_process in
                  observation_process_list]):
            observation_process = concatenateODESolution(
                    [observation_process.observations for observation_process in observation_process_list])
            observation_process = ContinuousObservationProcess(observation_process)

    return observation_process
