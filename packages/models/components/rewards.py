from abc import ABC, abstractmethod
from packages.types.spaces import Space, FiniteIntSpace
import torch
import torch.nn as nn


# Classes for reward models used in pomdps

class RewardModel(ABC, nn.Module):
    def __init__(self, SSpace: Space, ASpace: Space):
        """
        Abstract base class for a Reward Model
        :param SSpace: State space used in the Reward Model
        :param ASpace: Action space used in the Reward Model
        """
        self.SSpace = SSpace
        self.ASpace = ASpace
        nn.Module.__init__(self)

    @abstractmethod
    def __call__(self, state: torch.Tensor, action: torch.Tensor):
        """
        return reward (rate) for state and action
        """
        pass


class TabularRewardModel(RewardModel):
    def __init__(self, SSpace: FiniteIntSpace, ASpace: FiniteIntSpace, rewardMatrix: torch.Tensor):
        """
        Reward model defined by a table
        :param SSpace: State space used in the Reward Model
        :param ASpace: Action space used in the Reward Model
        :param rewardMatrix: np.array with table containing reward entries
        """
        super().__init__(SSpace, ASpace)
        reward_scale = rewardMatrix.abs().max()
        # Normalize Rewards
        rewardMatrix /= reward_scale
        self.register_buffer('reward_scale',reward_scale)
        self.register_buffer('rewardMatrix', rewardMatrix)

    def __call__(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.rewardMatrix[state, action]
