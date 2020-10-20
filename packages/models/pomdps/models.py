from typing import Union
import torch
from packages.types.spaces import Space, CountableSpace
from packages.models.components.transitions import TransitionModel
from packages.models.components.observations import ObservationModel
from packages.models.components.rewards import RewardModel
import torch.nn as nn


class POMDP(nn.Module):
    def __init__(self, SSpace: CountableSpace,
                 ASpace: CountableSpace,
                 OSpace: Space,
                 TModel: TransitionModel,
                 OModel: ObservationModel,
                 RModel: RewardModel,
                 discount: Union[torch.Tensor, float]):
        """
        POMDP class

        :param SSpace: The state space of the POMDP
        :param ASpace: The action space of the POMDP
        :param OSpace: The observation space of the POMDP
        :param TModel: The transition model of the POMDP
        :param OModel: The observation model of the POMDP
        :param RModel: The reward model of the POMDP
        :param discount: Discount factor of the POMDP
        """

        assert TModel.SSpace == SSpace
        assert TModel.ASpace == ASpace
        assert OModel.SSpace == SSpace
        assert OModel.ASpace == ASpace
        assert RModel.SSpace == SSpace
        assert RModel.ASpace == ASpace

        super().__init__()

        self.SSpace = SSpace
        self.ASpace = ASpace
        self.OSpace = OSpace
        self.TModel = TModel
        self.OModel = OModel
        self.RModel = RModel

        self.register_buffer('discount', torch.as_tensor(discount))

    def get_parameters(self):
        """
        Returns parameters of the pomdp that will be written to checkpoint files
        :return: Dictionary of parameters
        """
        params = {
            'discount': float(self.discount)
        }
        return params
