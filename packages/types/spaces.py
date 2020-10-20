from typing import Tuple, Union
from abc import ABC, abstractmethod
from packages.types.numbers import Number, RealNumbers, NaturalNumbers
import torch

# Classes for spaces used in models

class Space(ABC):
    @abstractmethod
    def __init__(self, domain: Number, dimension: int, isfinite: bool, iscountable: bool):
        """
        Abstract base class for a space, e.g. R^2
        :param domain: The domain of the space, e.g., reals or natural numbers
        :param dimension: The dimensionality of the space e.g. 2
        :param isfinite: boolean for finite spaces
        :param iscountable: boolean for countable spaces
        """
        # Save properties
        self.domain = domain
        self.dimension = dimension
        self.isfinite = isfinite
        self.iscountable = iscountable


class RealSpace(Space):
    def __init__(self, dimension: int = 1):
        """
        Class for R^d
        :param dimension: The dimensionality of the space, i.e., d for R^d
        """
        domain = RealNumbers()
        super().__init__(domain=domain, dimension=dimension, isfinite=False, iscountable=False)


class CountableSpace(Space, ABC):
    def __init__(self, dimension: int, isfinite: bool):
        domain = NaturalNumbers()
        super().__init__(domain=domain, dimension=dimension, isfinite=isfinite, iscountable=True)


class FiniteIntSpace(CountableSpace):
    def __init__(self, cardinalities: Union[torch.Tensor,int]):
        """
        Class for finite countable spaces
        :param cardinalities: cardinalities for each dimension (tuple of ints)
        """
        cardinalities=torch.as_tensor(cardinalities, dtype=torch.int)
        if cardinalities.ndim==0:
            cardinalities=cardinalities.view(-1)
        dimension = len(cardinalities)
        super().__init__(dimension=dimension, isfinite=True)
        self.cardinalities = cardinalities
        self.nElements=int(cardinalities.prod())


class InfiniteIntSpace(CountableSpace):
    """Class for infinite but countable Spaces"""

    def __init__(self, dimension=1):
        """

        :param dimension: The dimensionality of the space, i.e., d for N^d
        """
        super().__init__(dimension=dimension, isfinite=False)
