"""Acquisition function helpers for mille-feuille.

Provides factory functions that construct BOtorch acquisition functions
from a fitted surrogate and the current optimisation state.

"""

import torch
from botorch.acquisition import qLogExpectedImprovement, qUpperConfidenceBound
from botorch.acquisition.multi_objective.logei import (
    qLogExpectedHypervolumeImprovement,
)
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)

from .definitions import device, dtype


def get_qLogEI_acq(surrogate, state):
    """
    Build a qLogExpectedImprovement acquisition function for single-fidelity BO.

    Parameters
    ----------
    surrogate : BaseSurrogate
        A fitted single-fidelity surrogate.
    state : State
        Current optimisation state.

    Returns
    -------
    qLogExpectedImprovement
    """
    return qLogExpectedImprovement(surrogate.model, state.best_value_transformed)


def get_qUCB_acq(surrogate, state, beta: float = 1.0):
    """
    Build a qUpperConfidenceBound acquisition function for single-fidelity BO.

    Parameters
    ----------
    surrogate : BaseSurrogate
        A fitted single-fidelity surrogate.
    state : State
        Current optimisation state.
    beta : float, optional
        Exploration-exploitation trade-off parameter (default 1.0).

    Returns
    -------
    qUpperConfidenceBound
    """
    return qUpperConfidenceBound(surrogate.model, beta=beta)


def get_qLogEI_MF_acq(surrogate, state, weights=None):
    """
    Build a qLogExpectedImprovement acquisition function for multi-fidelity BO,
    using a ScalarizedPosteriorTransform to collapse the multi-output posterior
    to a single scalar.

    Parameters
    ----------
    surrogate : BaseSurrogate
        A fitted multi-fidelity surrogate.
    state : State
        Current optimisation state.
    weights : torch.Tensor or None, optional
        1-D weight tensor for the scalarization. Defaults to ``[1.0]``
        (selects the single target-fidelity output).

    Returns
    -------
    qLogExpectedImprovement
    """
    if weights is None:
        weights = torch.tensor([1.0], dtype=dtype, device=device)
    post_transform = ScalarizedPosteriorTransform(weights=weights)
    return qLogExpectedImprovement(
        surrogate.model,
        state.best_value_transformed,
        posterior_transform=post_transform,
    )


def get_qUCB_MF_acq(surrogate, state, beta: float = 1.0, weights=None):
    """
    Build a qUpperConfidenceBound acquisition function for multi-fidelity BO,
    using a ScalarizedPosteriorTransform to collapse the multi-output posterior
    to a single scalar.

    Parameters
    ----------
    surrogate : BaseSurrogate
        A fitted multi-fidelity surrogate.
    state : State
        Current optimisation state.
    beta : float, optional
        Exploration-exploitation trade-off parameter (default 1.0).
    weights : torch.Tensor or None, optional
        1-D weight tensor for the scalarization. Defaults to ``[1.0]``.

    Returns
    -------
    qUpperConfidenceBound
    """
    if weights is None:
        weights = torch.tensor([1.0], dtype=dtype, device=device)
    post_transform = ScalarizedPosteriorTransform(weights=weights)
    return qUpperConfidenceBound(surrogate.model, beta=beta, posterior_transform=post_transform)


def get_qLogEHVI_acq(surrogate, state, ref_point=None):
    """
    Build a qLogExpectedHypervolumeImprovement acquisition function for
    multi-objective BO.

    The reference point and ``NondominatedPartitioning`` are both computed in
    the *normalised* (model) output space - i.e. the space of the Y values
    returned by ``state.transform_XY()``.

    Parameters
    ----------
    surrogate : BaseSurrogate
        A fitted multi-objective surrogate (e.g. ``ModelListGP``).
    state : State
        Current optimisation state.  ``state.Ys`` must have shape
        ``(N, n_objectives)``.
    ref_point : array-like or None, optional
        Reference point in *normalised* output space, length ``n_objectives``.
        Defaults to ``state.worst_value_transformed`` (the per-objective minimum
        of observed data in normalised space), which is always dominated by all
        Pareto-optimal solutions seen so far.  The reference point must be
        dominated by all Pareto-optimal solutions.

    Returns
    -------
    qLogExpectedHypervolumeImprovement
    """
    _, Y_transformed = state.transform_XY()

    if ref_point is None:
        ref_point_t = torch.tensor(state.worst_value_transformed, dtype=dtype, device=device).flatten()
    else:
        ref_point_t = torch.tensor(ref_point, dtype=dtype, device=device)

    partitioning = FastNondominatedPartitioning(ref_point=ref_point_t, Y=Y_transformed)

    return qLogExpectedHypervolumeImprovement(
        model=surrogate.model,
        ref_point=ref_point_t,
        partitioning=partitioning,
    )
