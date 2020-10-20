import time
import logging
import torch
import torch.optim as optim
from torch import nn as nn
from torch.optim.optimizer import Optimizer
from typing import Optional
from abc import ABC, abstractmethod

from packages.models.pomdps.models import POMDP
from packages.types.spaces import FiniteIntSpace
from packages.models.components.advantages import AdvantageNet
from packages.models.components.values import ValueNet


class Learner(ABC, nn.Module):
    """ Learner classes are responsible for learning policies for a POMDP model using value and advantage networks. """

    def __init__(self, pomdp: POMDP, name: str, num_episodes: int, opt_constructor: Optional[Optimizer] = None,
                 optimizer_v_options=None, optimizer_a_options=None, advantage_net: AdvantageNet = None,
                 value_net: ValueNet = None,
                 checkpoint_interval: int = 50, discount_decay: int = 0):
        """
        :param pomdp: The POMDP to be solved
        :param name: Name of the learning process used for logging and checkpoints
        :param num_episodes: Number of episodes to train for
        :param device: The device used for training process
        :param opt_constructor: Optimizer used for network training
        :param optimizer_v_options: Options for the optimizer used in network training
        :param advantage_net: The advantage network used to represent the advantage function
        :param value_net: The value network used to represent the value function
        :param checkpoint_interval: Interval of epochs for saving checkpoint models. 0 will disable checkpoints.
        :param discount_decay: Decay for the discount parameter. 0 disables decay
        """
        ABC.__init__(self)
        nn.Module.__init__(self)

        self.logger = logging.getLogger(__name__)
        self.pomdp = pomdp
        self.name = name
        self.checkpoint_interval = checkpoint_interval
        self.num_episodes = num_episodes
        self.running_loss = 0
        self.discount_decay = float(discount_decay)

        try:
            assert isinstance(pomdp.ASpace, FiniteIntSpace)
        except:
            raise ValueError('Advantage updating can only be applied to discrete and finite action spaces.')

        # initialize optimizers
        if opt_constructor is None and optimizer_v_options is None:
            opt_constructor = optim.Adam
        if opt_constructor is None:
            opt_constructor = optim.Adam
        if optimizer_v_options is None:
            optimizer_v_options = dict()
        if optimizer_a_options is None:
            optimizer_a_options = dict()

        self.opt_constructor = opt_constructor
        self.optimizer_v_options = optimizer_v_options
        self.optimizer_a_options = optimizer_a_options

        # setup networks if not provided as parameters
        if advantage_net is None:
            assert isinstance(pomdp.SSpace, FiniteIntSpace)
            in_dim = pomdp.SSpace.nElements
            out_dim = pomdp.ASpace.nElements
            self.advantage_net = AdvantageNet(in_dim, out_dim)
        else:
            self.advantage_net = advantage_net

        if value_net is None:
            assert isinstance(pomdp.SSpace, FiniteIntSpace)
            in_dim = pomdp.SSpace.nElements
            self.value_net = ValueNet(in_dim)
        else:
            self.value_net = value_net

        self.optimizer_v = opt_constructor(self.value_net.parameters(), **self.optimizer_v_options)
        self.optimizer_a = opt_constructor(self.advantage_net.parameters(), **self.optimizer_a_options)

    def learn(self) -> [AdvantageNet, ValueNet]:
        """
        Runs the learning process
        :return: Pi: greedy policy parametrised by fitted advantage NN
                 advantage_net : fitted advantage NN
                 value_net: fitted value NN
        """
        self.running_loss = 0

        for e in range(self.num_episodes):
            # measure time for statistics
            start = time.time()

            data = self.train_episode(e)

            # save checkpoints
            if self.checkpoint_interval > 0 and (e % self.checkpoint_interval == 0 or e == self.num_episodes - 1):
                torch.save({
                    'value_net_state_dict': self.value_net.state_dict(),
                    'advantage_net_state_dict': self.advantage_net.state_dict(),
                    'data': data,
                    'pomdp_params': self.pomdp.get_parameters(),
                    'learner_params': self.get_parameters(),
                    'episode': e,
                }, "checkpoints/model_col_{}_{}".format(self.name, e))

            end = time.time()
            self.logger.info("Episode {} took {}s.".format(e, end - start))

    def get_decayed_log_discount(self, episode):
        if self.discount_decay > 0:
            d0 = torch.as_tensor(1e-4).log()
            d1 = self.pomdp.discount.log()
            discount = d1 + (d0 - d1) * (-episode / torch.as_tensor(self.discount_decay)).exp()
            return discount
        else:
            return self.pomdp.discount.log()

    def get_parameters(self):
        """
        Returns parameters of the learner that will be written to checkpoint files
        :return: Dictionary of parameters
        """
        params = {
            'discount_decay': self.discount_decay
        }
        return params

    @abstractmethod
    def train_episode(self, episode):
        """
        Excecutes a training episode
        :param episode: Number of episode that is run
        :return: the loss of the episode
        """
        raise NotImplementedError()
