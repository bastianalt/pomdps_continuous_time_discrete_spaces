import torch
import torch.nn as nn
from torch import sigmoid
from torch.nn.functional import relu

class AdvantageNet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """
        Deep Neural Net representing the advantage function. Input is a vector representing the (belief)state.
        Output is a [number of actions] vector containing the advantage values.

        :param in_dim: Number of input neurons (For advantage over belief --> size of the belief state vector)
        :param out_dim: Number of output neurons (total number of actions)
        """
        super(AdvantageNet, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear1 = nn.Linear(self.in_dim, self.in_dim)
        self.linear2 = nn.Linear(self.in_dim, self.in_dim)
        self.linear3 = nn.Linear(self.in_dim, self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the NN
        :param x: input tensor (belief vector)
        :return: advantage vector (advantage values for all actions)
        """
        h1 = sigmoid(self.linear1(x))
        h2 = sigmoid(self.linear2(h1))
        h3 = self.linear3(h2)
        advantage = h3 - h3.max(dim=-1,keepdim=True)[0]
        return advantage
