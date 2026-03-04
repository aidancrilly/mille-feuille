from .generators import (
    BayesianOptimisationGenerator,
    _greedy_exclusion,
    _probabilistic_threshold_filter,
)
from .optimise import *
from .simulator import *
from .surrogate import BaseSurrogate

"""
Defines some useful utility functions which do not fit into the defined classes
"""


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
    Samples points using a GP surrogate and filters them probabilistically
    based on the predicted probability of exceeding a threshold.

    Parameters:
        domain: InputDomain object
        state: State object
        sampler: QMC sampler (e.g. Sobol)
        surrogate: mille-feuille-compatible surrogate (must support predict(domain, X))
        initial_samples: number of samples to draw
        threshold_value: threshold value to compare against
        target_fidelity: int — selects surrogate fidelity
        target_key: int or str — selects surrogate output
        random_draws: optional np.ndarray of uniform(0,1) values of shape (initial_samples,)
                      if None, will be generated internally

    Returns:
        x_all: all sampled input points (shape: NxD)
        y_pred: np.ndarray of predicted means (selected column)
        prob: predicted P(y > threshold) for each point
        mask: boolean array of points that passed the stochastic filter
    """
    return _probabilistic_threshold_filter(
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
    Draws samples from the sampler, evaluates the surrogate model,
    and selects those above a specified threshold.

    Parameters:
        domain: InputDomain object
        state: State object
        sampler: QMC sampler (e.g., Sobol)
        surrogate: mille-feuille-style surrogate with .predict(domain, Xs) method
        initial_samples: int, number of candidate points
        surrogate_threshold: float, threshold to apply on predicted mean
        target_fidelity: int — selects surrogate fidelity
        target_key: int or str — selects surrogate output

    Returns:
        x_all: np.ndarray of all candidate input samples
        y_pred: np.ndarray of predicted means (selected column)
        mask: np.ndarray of boolean values where prediction > threshold
    """
    # Draw samples in unit hypercube, transform to input space
    x_unit = sampler.random(initial_samples)
    x_all = domain.inverse_transform(x_unit)

    # Predict mean and std from GP
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
    Samples points using a GP surrogate with probabilistic threshold filtering
    and proximity-based exclusion to prevent closely-clustered sample points.

    A large number of random sample points are drawn and filtered by the
    probabilistic threshold criterion. The remaining candidates are analysed
    for clusters; within each cluster, coordinates are rescaled via PCA to
    an uncorrelated unit-variance space. Points are then greedily selected
    up to batch_size, rejecting any candidate whose normalised distance to an
    already-selected point is less than rejection_radius.

    Parameters:
        domain: InputDomain object
        state: State object
        sampler: QMC sampler (e.g. Sobol)
        surrogate: mille-feuille-compatible surrogate (must support predict(domain, X))
        initial_samples: number of initial random candidates to draw
        threshold_value: threshold value for the probabilistic filter
        batch_size: maximum number of points to return
        rejection_radius: minimum distance in PCA-normalised space between selected points
        n_clusters: number of clusters used for PCA analysis (default: 1)
        target_fidelity: int — selects surrogate fidelity
        target_key: int or str — selects surrogate output
        random_draws: optional np.ndarray of uniform(0,1) values of shape (initial_samples,)
                      if None, will be generated internally

    Returns:
        x_selected: selected input points (shape: K x D, K <= batch_size)
        y_selected: predicted means at selected points (shape: K,)
        prob_selected: predicted P(y > threshold) at selected points (shape: K,)
    """
    x_all, y_pred, prob, mask = _probabilistic_threshold_filter(
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

    return _greedy_exclusion(
        x_candidates,
        y_candidates,
        prob_candidates,
        batch_size,
        rejection_radius,
        n_clusters,
    )


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
