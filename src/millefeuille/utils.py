import warnings

from .generators import BayesianOptimisationGenerator, greedy_exclusion, probabilistic_threshold_filter
import warnings

from .generators import BayesianOptimisationGenerator, greedy_exclusion, probabilistic_threshold_filter
from .optimise import *
from .simulator import *
from .surrogate import BaseSurrogate

"""
Utility functions for running generate-evaluate loops.

Candidate generation is handled by the generator classes in
``millefeuille.generators``.  This module provides loop runners that
pair any ``CandidateGenerator`` with a simulator.

The standalone sampling functions (``probabilistic_threshold_sampling``,
``surrogate_threshold_sampling``, ``probabilistic_threshold_sampling_with_exclusion``)
are kept for backwards compatibility but are deprecated — use the generator
classes in ``millefeuille.generators`` instead.
"""


# ---------------------------------------------------------------------------
# Deprecated standalone sampling helpers
# ---------------------------------------------------------------------------
Utility functions for running generate-evaluate loops.

Candidate generation is handled by the generator classes in
``millefeuille.generators``.  This module provides loop runners that
pair any ``CandidateGenerator`` with a simulator.

The standalone sampling functions (``probabilistic_threshold_sampling``,
``surrogate_threshold_sampling``, ``probabilistic_threshold_sampling_with_exclusion``)
are kept for backwards compatibility but are deprecated — use the generator
classes in ``millefeuille.generators`` instead.
"""


# ---------------------------------------------------------------------------
# Deprecated standalone sampling helpers
# ---------------------------------------------------------------------------


def probabilistic_threshold_sampling(
    domain,
    state,
    sampler,
    surrogate,
    initial_samples,
    threshold_value,
    target_fidelity=None,
    target_key=None,
    random_draws=None,
):
    """
    .. deprecated::
        Use :func:`millefeuille.generators.probabilistic_threshold_filter`
        or :class:`millefeuille.generators.ThresholdCandidateGenerator` instead.
    """
    warnings.warn(
        "probabilistic_threshold_sampling is deprecated — use "
        "millefeuille.generators.probabilistic_threshold_filter or "
        "ThresholdCandidateGenerator instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return probabilistic_threshold_filter(
    .. deprecated::
        Use :func:`millefeuille.generators.probabilistic_threshold_filter`
        or :class:`millefeuille.generators.ThresholdCandidateGenerator` instead.
    """
    warnings.warn(
        "probabilistic_threshold_sampling is deprecated — use "
        "millefeuille.generators.probabilistic_threshold_filter or "
        "ThresholdCandidateGenerator instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return probabilistic_threshold_filter(
        domain=domain,
        state=state,
        sampler=sampler,
        surrogate=surrogate,
        pool_size=initial_samples,
        threshold_value=threshold_value,
        target_fidelity=target_fidelity,
        target_key=target_key,
        random_draws=random_draws,
    )


def surrogate_threshold_sampling(
    domain, state, sampler, surrogate, initial_samples, surrogate_threshold, target_fidelity=None, target_key=None
):
    """
    .. deprecated::
        Use :class:`millefeuille.generators.SurrogateThresholdCandidateGenerator`
        instead.
    """
    warnings.warn(
        "surrogate_threshold_sampling is deprecated — use SurrogateThresholdCandidateGenerator instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    x_unit = sampler.random(initial_samples)
    x_all = domain.inverse_transform(x_unit)

    predictions = surrogate.predict(state, x_all)

    if target_key is not None:
        prediction = predictions[target_key]
    else:
        if target_fidelity is not None:
            prediction = predictions[target_fidelity]
        else:
            if "mean" not in predictions.keys():
                raise ValueError(
                    "mean missing from predictions.keys(), did you miss target_fidelity or target_key inputs?"
                )
            prediction = predictions

    mean = prediction["mean"]
    y_pred = mean
    mask = mean > surrogate_threshold

    return x_all, y_pred, mask


def probabilistic_threshold_sampling_with_exclusion(
    domain,
    state,
    sampler,
    surrogate,
    initial_samples,
    threshold_value,
    batch_size,
    rejection_radius,
    n_clusters=1,
    target_fidelity=None,
    target_key=None,
    random_draws=None,
):
    """
    .. deprecated::
        Use :class:`millefeuille.generators.ThresholdExclusionGenerator` or
        compose :class:`~millefeuille.generators.ThresholdCandidateGenerator`
        with :class:`~millefeuille.generators.GreedyExclusionGenerator` instead.
    """
    warnings.warn(
        "probabilistic_threshold_sampling_with_exclusion is deprecated — use "
        "ThresholdExclusionGenerator or compose ThresholdCandidateGenerator "
        "with GreedyExclusionGenerator instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    import numpy as np

    x_all, y_pred, prob, mask = probabilistic_threshold_filter(
    .. deprecated::
        Use :class:`millefeuille.generators.ThresholdExclusionGenerator` or
        compose :class:`~millefeuille.generators.ThresholdCandidateGenerator`
        with :class:`~millefeuille.generators.GreedyExclusionGenerator` instead.
    """
    warnings.warn(
        "probabilistic_threshold_sampling_with_exclusion is deprecated — use "
        "ThresholdExclusionGenerator or compose ThresholdCandidateGenerator "
        "with GreedyExclusionGenerator instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    import numpy as np

    x_all, y_pred, prob, mask = probabilistic_threshold_filter(
        domain=domain,
        state=state,
        sampler=sampler,
        surrogate=surrogate,
        pool_size=initial_samples,
        threshold_value=threshold_value,
        target_fidelity=target_fidelity,
        target_key=target_key,
        random_draws=random_draws,
    )

    x_candidates = x_all[mask]
    y_candidates = y_pred[mask]
    prob_candidates = prob[mask]

    if len(x_candidates) == 0:
        return x_candidates, y_candidates, prob_candidates

    # Sort by descending probability then apply exclusion
    order = np.argsort(-prob_candidates)
    x_sorted = x_candidates[order]
    y_sorted = y_candidates[order]
    prob_sorted = prob_candidates[order]

    selected_idx = greedy_exclusion(x_sorted, batch_size, rejection_radius, n_clusters)

    if len(selected_idx) == 0:
        return np.empty((0, x_all.shape[1])), np.empty(0), np.empty(0)

    return x_sorted[selected_idx], y_sorted[selected_idx], prob_sorted[selected_idx]


# ---------------------------------------------------------------------------
# Loop runners
# ---------------------------------------------------------------------------
    # Sort by descending probability then apply exclusion
    order = np.argsort(-prob_candidates)
    x_sorted = x_candidates[order]
    y_sorted = y_candidates[order]
    prob_sorted = prob_candidates[order]

    selected_idx = greedy_exclusion(x_sorted, batch_size, rejection_radius, n_clusters)

    if len(selected_idx) == 0:
        return np.empty((0, x_all.shape[1])), np.empty(0), np.empty(0)

    return x_sorted[selected_idx], y_sorted[selected_idx], prob_sorted[selected_idx]


# ---------------------------------------------------------------------------
# Loop runners
# ---------------------------------------------------------------------------


def run_Bayesian_optimiser(
    Nsamples,
    batch_size,
    generate_acq_function,
    state,
    surrogate,
    simulator,
    scheduler=None,
    csv_name=None,
    verbose=False,
    **kwargs,
):
    if isinstance(simulator, ExectuableSimulator) and scheduler is None:
        print("If simulator is an ExecutableSimulator, you must provide a scheduler")
        raise Exception
    assert isinstance(surrogate, BaseSurrogate)

    generator = BayesianOptimisationGenerator(
        domain=state.input_domain,
        surrogate=surrogate,
        generate_acq_fn=generate_acq_function,
        refit_surrogate=True,
        verbose=verbose,
        **kwargs,
    )

    return run_generator_loop(
        Nsamples=Nsamples,
        batch_size=batch_size,
        generate_candidates=generator,
        state=state,
        simulator=simulator,
        scheduler=scheduler,
        csv_name=csv_name,
    )


def run_generator_loop(
    Nsamples,
    batch_size,
    generate_candidates,
    state,
    simulator,
    scheduler=None,
    csv_name=None,
):
    """Synchronous generate-evaluate loop with a pluggable candidate generator.

    Generalises ``run_Bayesian_optimiser`` to accept any ``CandidateGenerator``
    instance or plain callable.  Each iteration generates *batch_size*
    candidates, evaluates them with the simulator, and updates the state.

    Parameters:
        Nsamples:           Number of generate-evaluate iterations.
        batch_size:         Candidates requested per iteration.
        generate_candidates:
            A ``CandidateGenerator`` instance or a callable with signature::

                generate_candidates(state, n) -> (indices, Xs, Ss | None)

            * *state* — current ``State``.
            * *n* — how many candidates are requested.
            * Returns ``indices`` (1-D int array), ``Xs`` (N x dim),
              and ``Ss`` (N x 1 or ``None``).

            If a ``CandidateGenerator`` is passed its ``__call__`` method
            is used directly.  A plain callable must return the same
            3-tuple.
        state:              Current ``State``.
        simulator:          ``ExectuableSimulator`` or ``PythonSimulator``.
        scheduler:          ``Scheduler`` instance (required for
                            ``ExectuableSimulator``).
        csv_name:           Optional CSV path to persist state after each
                            iteration.

    Returns:
        Updated ``State``.
    """
    if isinstance(simulator, ExectuableSimulator) and scheduler is None:
        print("If simulator is an ExecutableSimulator, you must provide a scheduler")
        raise Exception

    for _ in range(Nsamples):
        index_next, X_next, S_next = generate_candidates(state, batch_size)

        if isinstance(simulator, ExectuableSimulator):
            P_next, Y_next = simulator(index_next, X_next, scheduler, Ss=S_next)
        elif isinstance(simulator, PythonSimulator):
            P_next, Y_next = simulator(index_next, X_next, Ss=S_next)
        else:
            print("simulator class not recognised, inherit for mille-feuille Simulator classes...")

        state.update(index_next, X_next=X_next, Y_next=Y_next, P_next=P_next, S_next=S_next)
        if csv_name is not None:
            state.to_csv(csv_name)

    return state
