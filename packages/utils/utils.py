import random
from typing import List, Callable, Any, Union
import numpy as np
from torchdiffeq._impl.solvers import OdeTorchSolution
import torch

def piecewise(funclist: List[Union[Callable, np.ndarray]], bounds: np.ndarray, t: Any) -> np.ndarray:
    """
    returns the piecwise function evaluation at t, where bounds defines the time intervals for which funclist holds
    ,e.g., if t_i<=t<t_j and bounds[i,:]=t_i,t_j => return funclist[i], cf. np.piecwise
    :param funclist: functionlist containing callback functions
    :param bounds: matrix of bounds for the time intervals
    :param t: evaluation time
    :return: function evaluated at t
    """
    t = np.asarray(t)
    lower_bounds, upper_bounds = np.asarray(bounds).T

    logical_table = (lower_bounds.reshape(1, -1) <= t.reshape(-1, 1)) * (
            t.reshape(-1, 1) < upper_bounds.reshape(1, -1))
    idx_list = np.argmax(logical_table, axis=1)
    if t.size == 1:
        if callable(funclist[idx_list[0]]):
            out = funclist[idx_list[0]](t)
        else:
            out = funclist[idx_list[0]]
    else:
        out = [None for _ in idx_list.__len__()]
        for idx, ode_idx in enumerate(idx_list):
            if callable(funclist[ode_idx]):
                out[idx] = funclist[ode_idx](t[idx])
            else:
                # noinspection PyTypeChecker
                out[idx] = funclist[ode_idx]

        out = np.asarray(out)
    return out

def concatenateODESolution(ode_list: List[OdeTorchSolution]) -> OdeTorchSolution:
    """
    Concatenates a list of descending or ascending odes with dense output of type OdeTorchSolution
    :param ode_list: list of OdeTorchSolution objects
    :return: concatenated OdeTorchSolution
    """

    interpolants = sum([sol.interpolants for sol in ode_list], [])
    ts = torch.cat([sol.ts[:-1].view(-1) for sol in ode_list]+[ode_list[-1].ts[-1].view(-1)])
    ode = OdeTorchSolution(ts, interpolants, interpolator=ode_list[0].interpolator)
    return ode


def sampleDiscrete(weights, size=1, axis=0, keepdims=False, binsearch=True):
    """
    Generates samples from a set of discrete distributions.

    :param weights: Array of positive numbers representing the (unnormalized) weights of the distributions
    :param size: Integer indicating the number of samples to be generated per distribution
    :param axis: Axis along which the distributions are oriented in the array
    :param binsearch: If true, the distributions are processed sequentially but for each distribution the samples are
        drawn in parallel via binary search (fast for many categories and large numbers of samples). Otherwise, the
        distributions are processed in parallel but samples are drawn sequentially (fast for large number of
        distributions).
    :return: Array containing the samples. The shape coincides with that of the weight array, except that the length of
        the specified axis is now given by the size parameter.
    """
    # cast to numpy array and assert non-negativity
    weights = np.array(weights, dtype=float)
    try:
        assert np.all(weights >= 0)
    except AssertionError:
        raise ValueError('negative probability weights')

    # always orient distributions along the last axis
    weights = np.swapaxes(weights, -1, axis)

    # normalize weights and compute cumulative sum
    weights /= np.sum(weights, axis=-1, keepdims=True)
    csum = np.cumsum(weights, axis=-1)

    # get shape of output array and sample uniform numbers
    shape = (*weights.shape[0:-1], size)
    x = np.zeros(shape, dtype=int)
    p = np.random.random(shape)

    # generate samples
    if binsearch:
        # total number of distributions
        nDists = int(np.prod(weights.shape[0:-1]))

        # orient all distributions along a single axis --> shape: (nDists, size)
        csum = csum.reshape(nDists, -1)
        x = x.reshape(nDists, -1)
        p = p.reshape(nDists, -1)

        # generate all samples per distribution in parallel, one distribution after another
        for ind in range(nDists):
            x[ind, :] = np.searchsorted(csum[ind, :], p[ind, :])

        # undo reshaping
        x = x.reshape(shape)
    else:
        # generate samples in parallel for all distributions, sample after sample
        for s in range(size):
            x[..., s] = np.argmax(p[..., s] <= csum, axis=-1)

    # undo axis swapping
    x = np.swapaxes(x, -1, axis)

    # remove unnecessary axis
    if size == 1 and not keepdims:
        x = np.squeeze(x, axis=axis)
        if x.size == 1:
            x = int(x)

    return x


def bc2xy(p, corners):
    """
    Converts barycentric coordinates on a two-dimensional simplex to xy-Cartesian coordinates.

    Parameters
    ----------
    p: [N x 3] array representing points on the two-dimensional probability simplex
    corners: [3 x 2] array whose rows specify the three corners of the simplex

    Returns
    -------
    [N x 2] array containing the xy-Cartesian coordinates of the points
    """
    assert p.shape[1] == 3
    assert corners.shape == (3, 2)
    return p @ corners


def xy2bc(xy, corners):
    """
    Converts xy-Cartesian coordinates to barycentric coordinates on a two-dimensional simplex.

    Parameters
    ----------
    xy: [N x 2] array containing the xy-Cartesian coordinates of the points
    corners: [3 x 2] array whose rows specify the three corners of the simplex

    Returns
    -------
    [N x 3] array representing points on the two-dimensional probability simplex
    """
    A = np.c_[corners, np.ones(3)]
    b = np.c_[xy, np.ones(xy.shape[0])]
    p = np.clip(np.linalg.solve(A.T, b.T).T, 0, 1)
    return p


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, element):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = element
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)