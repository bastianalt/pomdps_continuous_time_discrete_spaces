import torch
import torch.nn as nn
from torch import sigmoid
from typing import Optional
from torch.nn.functional import relu, threshold


class ValueNet(nn.Module):
    def __init__(self, in_dim: int):
        """
          Base class for Deep Neural Net representing the value function.

              :param in_dim: Number of input neurons (For Value over belief --> size of the belief state vector)
        """
        nn.Module.__init__(self)
        self.in_dim = in_dim
        self.linear1 = nn.Linear(self.in_dim, self.in_dim)
        self.linear2 = nn.Linear(self.in_dim, self.in_dim)
        self.linear3 = nn.Linear(self.in_dim, 1)

    def forward(self, x: torch.Tensor, compute_grad=True, compute_hessian=False) -> [torch.Tensor,
                                                                                     Optional[torch.Tensor],
                                                                                     Optional[torch.Tensor]]:
        """
              Forward pass through the NN
              :param x: input tensor (belief vector)
              :param compute_grad: Boolean if gradient of value function is computed
              :param compute_hessian: Boolean if Hessian of value function is computed
              :return: value: value for the input
                       grad: gradient of the input
                       hessian: hessian of the input
        """
        a1 = self.linear1(x)
        h1 = sigmoid(a1)

        a2 = self.linear2(h1)
        h2 = sigmoid(a2)

        value = self.linear3(h2)

        if compute_grad:

            aj1 = sigmoid_derivative(a1)
            m1 = self.linear1.weight[None, ...]
            # j1 = torch.diag(aj1) @ m1
            j1 = aj1[..., None] * m1

            aj2 = sigmoid_derivative(a2)
            m2 = self.linear2.weight[None, ...] @ j1
            # j2 = torch.diag(aj2) @ m2
            j2 = aj2[..., None] * m2

            m3 = self.linear3.weight[None, ...] @ j2
            grad = m3.squeeze(dim=1)

            if compute_hessian:
                jj1 = sigmoid_2ndderivative(a1)[..., None, None] * m1[..., None] * m1[:, :, None, :]

                jj2 = sigmoid_2ndderivative(a2)[..., None, None] * m2[..., None] * m2[:, :, None, :] + aj2[
                    ..., None, None] * torch.sum(self.linear2.weight[None, ..., None, None] * jj1[:, None, ...],
                                                 dim=2)

                hessian = torch.sum(self.linear3.weight[None, ..., None, None] * jj2[:, None, ...], dim=2).squeeze(
                        dim=1).squeeze(dim=0)
                return value, grad, hessian
            else:
                return value, grad
        else:
            return value


def relu_derivative(x: torch.Tensor) -> torch.Tensor:
    return threshold(x, 0.0, 1.0)


def relu_2ndderivative(x: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(x)


def sigmoid_derivative(x: torch.Tensor) -> torch.Tensor:
    """
    Helper function representing the derivative of the sigmoid function
    """
    return sigmoid(x) * (torch.ones_like(x) - sigmoid(x))


def sigmoid_2ndderivative(x: torch.Tensor) -> torch.Tensor:
    """
    Helper function representing the 2nd derivative of the sigmoid function
    """
    return sigmoid(x) * (torch.ones_like(x) - sigmoid(x)) * (torch.ones_like(x) - 2 * sigmoid(x))


class ValueNet2(nn.Module):
    def __init__(self, in_dim):
        """
        Value function representation as neural network
        :param in_dim: Number of input neurons (For Value over belief --> size of the belief state vector)
        """
        super(ValueNet2, self).__init__()

        self.in_dim = in_dim

        self.layers = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor, compute_grad=True, compute_hessian=False) -> [torch.Tensor,
                                                                                     Optional[torch.Tensor],
                                                                                     Optional[torch.Tensor]]:
        """
              Forward pass through the NN
              :param x: input tensor (belief vector)
              :param compute_grad: Boolean if gradient of value function is computed
              :param compute_hessian: Boolean if Hessian of value function is computed
              :return: value: value for the input
                       grad: gradient of the input
                       hessian: hessian of the input
        """
        if compute_grad:
            torch.set_grad_enabled(True)
            x.requires_grad = True
            #x = torch.autograd.Variable(x, requires_grad=True)

        # reshape to bach x channels
        x_batch = x.flatten(start_dim=0, end_dim=-2)

        # apply network
        out = self.layers(x_batch)

        # reshape again to input shape
        value = out.view(*x.shape[:-1], -1)

        if compute_grad:
            grad = torch.autograd.grad(torch.sum(value), x, create_graph=True)[0]
            return value, grad

        if compute_hessian:
            raise AttributeError('Computing Hessian is currently not supported.')

        return value


class DeepValueNet(nn.Module):
    def __init__(self, to_grid_fun):
        """
          Value function representation as deep neural network
          :param to_grid_fun: Function that converts states to a grid that the CNN operates on
        """
        super(DeepValueNet, self).__init__()

        #self.shape = shape
        #self.num_s = num_s
        self.to_grid_fun = to_grid_fun

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(16, 4),
            nn.Sigmoid(),
            nn.Linear(4, 1)
        )

    # Defining the forward pass
    def forward(self, x_in: torch.Tensor, compute_grad=True, compute_hessian=False):
        if compute_grad:
            x_in.requires_grad = True

        # reshape to grid
        x_grid = self.to_grid_fun(x_in, walls_fill=0.)

        # reshape to bach x channels
        x_batch = x_grid.flatten(start_dim=0, end_dim=-3)[:, None]
        #x_batch = x_grid.view(-1, 1, x_grid.shape[-2], x_grid.shape[-1])

        # apply network
        x = self.cnn_layers(x_batch)
        x = x.view(x.size(0), -1)
        out = self.linear_layers(x)

        # reshape again to input shape
        value = out.view(*x_grid.shape[:-2], -1)

        if compute_grad:
            grad = torch.autograd.grad(torch.sum(value), x_in, create_graph=True)[0]
            return value, grad

        if compute_hessian:
            raise AttributeError('Computing Hessian is currently not supported.')

        return value
