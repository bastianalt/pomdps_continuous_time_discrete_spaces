from abc import ABC
import torch
from packages.models.components.transitions import TabularTransitionModel
from packages.models.components.observations import RandomDiscreteDensityObservationModel, ContinuousObservationModel
import torch.nn as nn
from torch.distributions.categorical import Categorical


# Classes for dynamics used in filters, Losses, ...

class Dynamics(ABC, nn.Module):
    def __init__(self, **kwargs):
        """
        Base class for a dynamical model
        """
        super().__init__(**kwargs)


class PiecewiseDeterministicMarkov(Dynamics):
    def __init__(self, TModel: TabularTransitionModel, OModel: RandomDiscreteDensityObservationModel):
        """
        Piecewise deterministic markov dynamic describing the dynamics of a continuous discrete filter
        :param TModel: Tabular transition model used for the prior dynamics
        :param OModel: Discrete observation model used for jumps
        """
        super().__init__()

        self.TModel = TModel
        self.OModel = OModel

        self.register_buffer('transition_matrix', self.TModel.transitionMatrix)


class WonhamDiffusion(Dynamics):
    def __init__(self, TModel: TabularTransitionModel, OModel: ContinuousObservationModel, **kwargs):
        """
        Diffusion Dynamics of the Wonham filter
        :param TModel: Tabular transition model used for the prior dynamics
        :param OModel: Continuous observation model used for Wonham dynamics
        """
        super().__init__(**kwargs)

        self.TModel = TModel
        self.OModel = OModel

        self.register_buffer('transition_matrix', self.TModel.transitionMatrix)  # S' S A
        self.register_buffer('drift_tensor', self.OModel.drift_tensor.permute(2, 1, 0))  # A S O
        self.register_buffer('dispersion_matrix', self.OModel.dispersion_matrix)
        self.register_buffer('outer_dispersion_inv', torch.inverse(self.dispersion_matrix @ self.dispersion_matrix.T))

    # noinspection SpellCheckingInspection,SpellCheckingInspection,SpellCheckingInspection,SpellCheckingInspection
    def diffusion_dynamics(self, beliefs: torch.Tensor, actions=None, compute_drift=True, compute_dispersion=True,
                           approx_state_exp_sampling=False):
        """
        Computes the Wonham diffusion dynamics
        :param beliefs: Tensor of beliefs. Must be of shape [Batch x States]
        :param actions: List of actions to consider of shape [Batch x Actions]. If None, all actions will be considered.
        :param compute_drift: Bool whether the drift terms should be computed
        :param compute_dispersion: Bool whether the drift terms should be computed
        :param approx_state_exp_sampling: If True, expectation over next state is approximated by sampling one state
        :return: Drift vector of shape [Batch x Actions x States] if compute_drift is True,
                 Dispersion matrix of shape [Batch, Actions, States, Noise] if compute_dispersion is True
        """
        if actions is None:
            actions = torch.arange(self.drift_tensor.shape[0])[None]
        elif actions.ndim == 1:
            actions = actions.reshape(-1, 1)

        # g_bar is of shape [N=NumSamples, ADim, ODim]
        da = self.drift_tensor[actions]  # drift tensor of selected actions: B A S O
        # noinspection PyArgumentList
        g_bar = torch.sum(beliefs[:, None, :, None] * da, axis=2)

        # g minus g_bar is of shape [N, A, S, O]
        gmgb = da - g_bar[:, :, None, :]

        # inv(hh) * (g-g_bar) is of shape [NumSamples, A, S, O]
        hhg = gmgb @ self.outer_dispersion_inv

        result = []

        if compute_drift:
            # GHG is of shape B, A, S
            # noinspection PyArgumentList
            ghg = torch.sum(gmgb * hhg, axis=-1)

            # BGHG is of shape B, A, S
            bghg = beliefs[:, None, :] * ghg

            # sumtb is of shape B, A, S
            if not approx_state_exp_sampling:
                t = self.transition_matrix[..., actions].permute(2, 3, 0, 1)  # B A S' S'
                # noinspection PyArgumentList
                tb = torch.sum(beliefs[:, None, :, None] * t, axis=2)
            else:
                ss = Categorical(probs=beliefs).sample()[:, None]  # B x 1
                tb = self.transition_matrix[ss, :, actions]

            # mu is of shape N, A, S
            mu_vec = bghg + tb

            result += [mu_vec]

        if compute_dispersion:
            # of shape N, A, S, NoiseDim
            sigma = beliefs[:, None, :, None] * torch.tensordot(self.dispersion_matrix, hhg,
                                                                dims=([0], [-1])).permute(1, 2, 3, 0)

            result += [sigma]

        if len(result) == 1:
            return result[0]
        else:
            return result
