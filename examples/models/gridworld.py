import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from itertools import groupby, cycle
from packages.models.pomdps.models import POMDP
from packages.types.spaces import FiniteIntSpace
from packages.models.components.transitions import TabularTransitionModel
from packages.models.components.observations import RandomFiniteObservationModel
from packages.models.components.rewards import TabularRewardModel
import torch.distributions
import torch


class Gridworld(POMDP):
    """Partial-observable grid world example."""

    def __init__(self, walls=None, goal=-1, discount=.9, observation_rate=10, transition_rate=3, transition_p_correct=-1):
        """
        :param walls: Boolean tensor indicating walls (True) and the size of the world (the shape of the tensor)
        :param goal: The goal field (field id, not it's position).
        :param discount: The discount factor
        :param observation_rate: The observation rate
        :param transition_rate: The positive transition rate if it should be fixed for all fields. If it is negative,
                                the rate is not fixed (so from fields at the borders can have a lower transition rate)
                                but is scaled with the absolute value. No scaling = -1
        :param transition_p_correct: The probability (between 0 and 1) to transition in the indicated direction.
                                     Probability mass that is left (1-p) will be divided up for the other directions.
        :param n_actions: Number of actions. 4 for all actions (down, right, up, left), 2 for only horizontal actions
                                             (right, left)
        """

        # set walls that also define the state space
        if walls is None:
            walls = torch.zeros(6, 6, dtype=bool)
            walls[2, 0:4] = 1
            if goal == -1:
                goal = [3, 2]
        self.goal = goal
        self.walls = walls

        # initialize objects
        # state space
        n_states = self.n_states
        s_space = FiniteIntSpace(n_states)

        # action space
        self.n_actions = 4
        a_space = FiniteIntSpace(self.n_actions)

        # transition model
        transition_matrix = self.construct_transistion_matrix(transition_rate, transition_p_correct)
        t_model = TabularTransitionModel(s_space, a_space, transition_matrix)

        # observation model
        n_observations = self.n_states
        o_space = FiniteIntSpace(n_observations)

        observation_tensor = self.construct_noisy_observation_matrix(sigma=.1)[..., None].repeat(1, 1, self.n_actions)
        o_model = RandomFiniteObservationModel(s_space, a_space, o_space, observation_rate, observation_tensor)

        # Reward Model
        # the goal is to be in state 0
        reward = 5
        reward_matrix = torch.zeros((n_states, self.n_actions))
        goal = torch.as_tensor(goal)
        if goal.shape[-1] == 1:
            reward_matrix[goal, :] = reward
        elif goal.shape[-1] == 2:
            reward_matrix[self.state_by_grid_position(goal), :] = reward
        else:
            raise AttributeError("Unknown goal attribute.")
        r_model = TabularRewardModel(s_space, a_space, reward_matrix)

        # POMDP model
        super().__init__(s_space, a_space, o_space, t_model, o_model, r_model, discount)

    @property
    def shape(self):
        return self.walls.shape if self.walls is not None else None

    @property
    def height(self):
        return self.walls.shape[0]

    @property
    def width(self):
        return self.walls.shape[1]

    @property
    def size(self):
        return torch.prod(torch.as_tensor(self.shape)).item()

    @property
    def n_walls(self):
        return self.walls.sum() if self.walls is not None else None

    @property
    def n_states(self):
        return self.size - self.n_walls

    @property
    def state_position(self):
        """
        Returns grid positions of all valid states
        :return: Tensor of grid positions for all valid state indices, shape: n_states x 2
        """
        if self.walls is None:
            return None
        else:
            gridIndices = np.c_[np.unravel_index(range(self.size), self.shape)]
            return torch.as_tensor(gridIndices[~self.walls.flatten(), :])

    def state_by_grid_position(self, p):
        """
        Returns the state index given a grid position
        :param p: the grid position or a batch of grid positions
        :return: the corresponding state indices
        """
        p = torch.as_tensor(p)
        assert p.shape[-1] == 2

        aa = torch.arange(self.size) - torch.cumsum(self.walls.flatten(), 0)
        aa[self.walls.flatten()] = -1
        aa = torch.reshape(aa, self.shape)
        bb = aa[p[..., 0], p[..., 1]]

        return bb

    def reshape_states_to_grid(self, states, walls_fill=-1):
        """
        Reshapes a tensor with states in the last dimension to a tensor of the grid
        :param states: tensor of values with number of states as the last dimension
        :param walls_fill: value that is used to fill resulting tensor at wall positions
        :return: the grid tensor
        """
        assert(states.shape[-1] == self.n_states)

        # create grid
        grid = torch.empty(*states.shape[:-1], self.height, self.width, device=states.device)

        grid_flat = torch.flatten(grid, -2, -1)
        walls_flat = torch.flatten(self.walls)

        # set walls' and states' values
        grid_flat[..., walls_flat] = walls_fill
        grid_flat[..., ~walls_flat] = states

        return grid

    def grid_position(self, s):
        """
        Returns for states the position on the grid
        :param s: the grid position or a tensor of grid positions
        :return: the resulting
        """
        s = torch.as_tensor(s)

        cum_walls = torch.cumsum(self.walls.flatten(), 0)
        s += torch.index_select(cum_walls, 0, s).reshape_as(s)
        x = s % self.shape[1]
        y = s // self.shape[1]

        res = torch.stack((y, x), axis=-1)
        return res

    def construct_transistion_matrix(self, transition_rate, p_correct=-1.):
        if p_correct <= 0:
            p_correct = .7
        elif p_correct > 1:
            raise AttributeError("p_correct must be beween 0 and 1")

        n_fields = self.size
        n_states = self.n_states

        # create a map where only one field is 1
        mp_grid = torch.eye(n_fields).reshape(n_fields, self.height, self.width)

        # convolute motion patterns with map
        basic_pattern = torch.zeros(3, 3)
        basic_pattern[0, 1] = p_correct
        basic_pattern[1, [0, 2]] = (1 - p_correct)/3
        basic_pattern[2, 1] = (1 - p_correct) / 3
        motion_patterns = torch.stack([torch.rot90(basic_pattern, r) for r in range(4)])
        tm = torch.conv2d(input=mp_grid[:, None], weight=motion_patterns[:, None, ...], padding=[1, 1])
        assert(tm.shape[-2:] == mp_grid.shape[-2:])

        # flatten map and extract valid fields
        tm_flat = torch.flatten(tm, 2, 3).permute(2, 0, 1)  # s' s a
        valid_fields = torch.flatten(~self.walls)
        tm_flat_valid = tm_flat[valid_fields][:, valid_fields]
        assert(tm_flat_valid.shape[0] == tm_flat_valid.shape[1] and tm_flat_valid.shape[0] == n_states)

        # remove self transitions
        self_trans = torch.eye(n_states, dtype=torch.bool).flatten()
        tm_flat_valid_flat = torch.flatten(tm_flat_valid, 0, 1)
        tm_flat_valid_flat[self_trans] = 0

        # renormalize to transition rate
        if transition_rate > 0:
            tm_norm = transition_rate * tm_flat_valid / torch.sum(tm_flat_valid, axis=0, keepdim=True)
        else:
            tm_norm = - tm_flat_valid * transition_rate  # if trainsition rate negative it is just a multiplier
        tm_norm -= torch.eye(n_states)[..., None] * torch.sum(tm_norm, axis=0, keepdim=True)  # set self transition rate

        return tm_norm

    def generate_Normal_2d(self, s, sigma=1):
        sdim_sigma = False
        if not torch.is_tensor(sigma) or sigma.ndim == 0:
            sigma = torch.as_tensor(sigma).flatten()
            sdim_sigma = True

        mid = torch.as_tensor([s-1, s-1], dtype=torch.float) / 2

        x = torch.arange(s).type(torch.float)
        mx = torch.stack(torch.meshgrid(x, x)).permute(1, 2, 0)

        # Gaussian PDF with symmetric density
        pds = -torch.sum((mx - mid[None, None, :]) ** 2, axis=-1)[None, ...] / (2 * sigma[:, None, None] ** 2)
        pds = pds.exp()
        pds /= torch.sum(pds, axis=[1, 2], keepdim=True)

        if sdim_sigma:
            return pds[0]
        else:
            return pds

    def generate_Log_Normal_2d(self, s):
        sigma = 1
        sdim_sigma = False
        if not torch.is_tensor(sigma) or sigma.ndim == 0:
            sigma = torch.as_tensor(sigma).flatten()  #TODO accept matrices evtl
            sdim_sigma = True

        mid = torch.as_tensor([s-1, s-1], dtype=torch.float) / 2

        x = torch.arange(s).type(torch.float)
        mx = torch.stack(torch.meshgrid(x, x)).permute(1, 2, 0)

        # Gaussian PDF with symmetric density
        pds = -torch.sum((mx - mid[None, None, :]) ** 2, axis=-1)[None, ...]
        return pds

    def construct_noisy_observation_matrix(self, sigma=1., filter_half_width=2):
        n_fields = self.size
        n_states = self.n_states

        # create a map where only one field is 1 in each map
        mp_grid = torch.eye(n_fields).reshape(n_fields, self.walls.shape[0], self.walls.shape[1])

        # convolute observation patterns with map
        filter = self.generate_Normal_2d(2*filter_half_width+1, sigma)
        om = torch.conv2d(input=mp_grid[:, None], weight=filter[None, None], padding=[filter_half_width, filter_half_width])
        assert(om.shape[-2:] == mp_grid.shape[-2:])

        # flatten map again and extract valid fields
        om_flat = torch.flatten(om, 2, 3).permute(2, 0, 1)  # o s a
        valid_fields = torch.flatten(~self.walls)
        om_flat_valid = om_flat[valid_fields][:, valid_fields]
        assert(om_flat_valid.shape[0] == om_flat_valid.shape[1] and om_flat_valid.shape[0] == n_states)

        # normalize
        om_norm = om_flat_valid[:, :, 0]
        om_norm /= torch.sum(om_norm, axis=0, keepdim=True)  # renormalize to transition rate

        return om_norm

    def get_parameters(self):
        params = super().get_parameters()
        params['walls'] = self.walls
        params['goal'] = self.goal
        return params


class GridWorldVisualization:
    def __init__(self, world: Gridworld):
        self.world = world

    def plot(self, values=None, cmap=None):
        fig, ax = plt.subplots()

        if values is None:
            if cmap is None:
                cmap = plt.get_cmap('gray').reversed()
            pltmap = self.world.walls.type(torch.float)
            pltmap[self.world.goal[0], self.world.goal[1]] = 0
            plt.imshow(pltmap, cmap=cmap, vmin=0, vmax=1)
        elif np.ndim(values) == 2:
            im = np.reshape(values, (*self.world.shape, -1))
            plt.imshow(im, cmap=cmap)
        else:
            if cmap is None:
                cmap = plt.get_cmap('viridis')
                cmap.set_bad(color='k')
            map = np.full(self.world.shape, np.nan)
            map[tuple(self.world.state_position.T)] = values
            plt.imshow(map, cmap=cmap)
        ax.set_xticks(np.arange(self.world.shape[1]) - 0.5, minor=True)
        ax.set_xticks(np.arange(self.world.shape[1]), minor=False)
        ax.xaxis.grid(True, which='minor', linewidth=2)
        ax.xaxis.grid(False, which='major')
        ax.set_yticks(np.arange(self.world.shape[0]) - 0.5, minor=True)
        ax.set_yticks(np.arange(self.world.shape[0]), minor=False)
        ax.yaxis.grid(True, which='minor', linewidth=2)
        ax.yaxis.grid(False, which='major')

    def plot_trajectories(self, trajectories):
        # iterate over all trajectories
        for trajectory in trajectories:
            # remove duplicate successive states
            trajectory = [s for s, _ in groupby(trajectory)]

            # get 2d positions of trajectory states
            positions = self.world.state_position[trajectory, :]

            # linear length along the points
            distance = np.cumsum(np.sqrt(np.sum(np.diff(positions, axis=0) ** 2, axis=1)))
            distance = np.insert(distance, 0, 0) / distance[-1]

            # interpolator object
            interpolator = interp1d(distance, positions, kind='quadratic', axis=0)

            # interpolation grid and interpolated values
            grid = np.linspace(0, 1, 10 * len(trajectory) - 1)
            interpolation = interpolator(grid)

            # plot the interpolated line
            plt.plot(interpolation[:, 1], interpolation[:, 0], color='k', linewidth=3, zorder=5)
            plt.plot(interpolation[:, 1], interpolation[:, 0], color='yellow', linewidth=1.2, zorder=5)

    def plot_arrows_xy(self, y, x, mags=None, length=0.5, head_width=0.2, head_length=0.2):
        # if no magnitudes specified, use constant magnitude of 1
        if mags is None:
            mags = cycle([1])

        inds = np.arange(y.shape[0])

        # norm x, y
        norm = torch.sqrt(x ** 2 + y ** 2)
        norm[norm < 1e-5] = 1e-5
        x = x/norm
        y = y/norm

        # plot all arrows
        for pos, ind, mag in zip(self.world.state_position, inds, mags):
            if mag == 0:
                continue
            dx = length * mag * x[ind]
            dy = -length * mag * y[ind]
            plt.arrow(pos[1] - dx / 2, pos[0] - dy / 2, dx, dy,
                      head_width=1.4 * mag * head_width, head_length=mag * head_length,
                      length_includes_head=True, facecolor='white', edgecolor='k', width=0.08)

    def plot_advantages(self, advantages):
        # advantages are of shape [states, actions]
        if advantages.shape[1] == 2:
            y = torch.zeros(advantages.shape[0])
            x = advantages[:, 0] - advantages[:, 1]
        elif advantages.shape[1] == 4:
            y = advantages[:, 2] - advantages[:, 0]
            x = advantages[:, 1] - advantages[:, 3]
        else:
            raise AttributeError()
        self.plot_arrows_xy(y, x)

    def plot_values(self, values, walls_fill=.1):
        fig, ax = plt.subplots()

        pltmap = self.world.reshape_states_to_grid(values, walls_fill)
        plt.imshow(pltmap, vmin=values.min(), vmax=values.max())

        ax.set_xticks(np.arange(self.world.shape[1]) - 0.5, minor=True)
        ax.set_xticks(np.arange(self.world.shape[1]), minor=False)
        ax.xaxis.grid(True, which='minor', linewidth=2, zorder=0)
        ax.xaxis.grid(False, which='major')
        ax.set_yticks(np.arange(self.world.shape[0]) - 0.5, minor=True)
        ax.set_yticks(np.arange(self.world.shape[0]), minor=False)
        ax.yaxis.grid(True, which='minor', linewidth=2)
        ax.yaxis.grid(False, which='major')

        plt.tick_params(
            axis='both',  # changes apply to the x-axis
            which='minor',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            left=False,
            right=False,
            labelbottom=False)  # labels along the bottom edge are off

        return pltmap

    def plot_walls(self, color='k'):
        walls = torch.stack(torch.where(self.world.walls)).T
        for w in walls:
            offs = .5
            plt.fill_between([w[1]-offs, w[1]+offs], w[0]-offs, w[0]+offs, color=color)
