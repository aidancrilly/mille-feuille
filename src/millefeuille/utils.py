import logging

import numpy as np
from scipy.stats import norm

from .optimise import *
from .asynch import AsyncScheduler, ResourceManager
from .generators import BayesianOptimisationGenerator, CandidateGenerator
from .simulator import *
from .surrogate import BaseSurrogate

logger = logging.getLogger("millefeuille.scheduler")

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
    # Generate inputs
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

    mean, std = prediction["mean"], prediction["std"]

    # Compute probability P(y > threshold)
    mean = mean.flatten()
    std = std.flatten()
    prob = 1.0 - norm.cdf(threshold_value, loc=mean, scale=std)

    # Generate random draws
    if random_draws is None:
        random_draws = np.random.rand(initial_samples)

    mask = prob > random_draws
    y_pred = mean

    return x_all, y_pred, prob, mask


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
    # Step 1: Draw candidates and apply probabilistic threshold filter
    x_all, y_pred, prob, mask = probabilistic_threshold_sampling(
        domain,
        state,
        sampler,
        surrogate,
        initial_samples,
        threshold_value,
        target_fidelity=target_fidelity,
        target_key=target_key,
        random_draws=random_draws,
    )

    x_candidates = x_all[mask]
    y_candidates = y_pred[mask]
    prob_candidates = prob[mask]

    if len(x_candidates) == 0:
        return x_candidates, y_candidates, prob_candidates

    n_candidates, n_dims = x_candidates.shape

    # Step 2: Cluster analysis — fall back gracefully if too few points
    n_clusters_actual = min(n_clusters, n_candidates)
    if n_clusters_actual > 1:
        from scipy.cluster.vq import kmeans2

        _, labels = kmeans2(x_candidates, n_clusters_actual, minit="points", seed=0)
    else:
        labels = np.zeros(n_candidates, dtype=int)
        n_clusters_actual = 1

    # Step 3: PCA-normalise within each cluster
    x_normalised = np.empty_like(x_candidates)
    for k in range(n_clusters_actual):
        cluster_mask = labels == k
        x_cluster = x_candidates[cluster_mask]

        if len(x_cluster) == 0:
            continue

        mean = x_cluster.mean(axis=0)
        x_centered = x_cluster - mean

        if len(x_cluster) == 1 or n_dims == 1:
            # Scalar normalisation along each dimension
            std = x_centered.std(axis=0)
            std = np.where(std > 0, std, 1.0)
            x_normalised[cluster_mask] = x_centered / std
        else:
            cov = np.cov(x_centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            eigenvalues = np.maximum(eigenvalues, 1e-10)
            x_normalised[cluster_mask] = x_centered @ eigenvectors / np.sqrt(eigenvalues)

    # Step 4: Greedy selection with rejection based on distance in normalised space.
    # Exclusion is applied within each cluster independently, since the PCA
    # normalised coordinates have different origins and scales per cluster.
    # Prioritise highest-probability candidates.
    sort_idx = np.argsort(-prob_candidates)

    selected_indices = []
    # Map cluster label -> list of already-selected normalised coordinates in that cluster
    selected_normalised_per_cluster = {k: [] for k in range(n_clusters_actual)}

    for idx in sort_idx:
        if len(selected_indices) >= batch_size:
            break

        cluster_k = labels[idx]
        x_norm_i = x_normalised[idx]
        already_selected = selected_normalised_per_cluster[cluster_k]

        if len(already_selected) == 0:
            selected_indices.append(idx)
            already_selected.append(x_norm_i)
        else:
            dists = np.linalg.norm(np.array(already_selected) - x_norm_i, axis=1)
            if np.all(dists >= rejection_radius):
                selected_indices.append(idx)
                already_selected.append(x_norm_i)

    if len(selected_indices) == 0:
        return np.empty((0, n_dims)), np.empty(0), np.empty(0)

    selected_indices = np.array(selected_indices)
    return x_candidates[selected_indices], y_candidates[selected_indices], prob_candidates[selected_indices]


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

    for _ in range(Nsamples):
        surrogate.fit(state)
        acq_function = generate_acq_function(surrogate, state)

        if state.l_MultiFidelity:
            X_next, S_next = suggest_next_locations(
                batch_size, state, acq_function=acq_function, verbose=verbose, **kwargs
            )
        else:
            X_next = suggest_next_locations(batch_size, state, acq_function=acq_function, verbose=verbose, **kwargs)
            S_next = None

        index_next = state.index[-1] + np.arange(batch_size) + 1
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


def run_async_Bayesian_optimiser(
    total_evaluations,
    batch_size,
    generate_acq_function,
    state,
    surrogate,
    simulator,
    resource_manager,
    scheduler=None,
    fidelity_configs=None,
    retrain_interval=None,
    max_workers=16,
    poll_interval=0.5,
    csv_name=None,
    verbose=False,
    **kwargs,
):
    """Asynchronous Bayesian optimisation loop.

    Replaces the synchronous *batch -> wait -> retrain* cycle with
    continuous scheduling: jobs are launched as soon as cores become
    available, and the surrogate is retrained after every
    *retrain_interval* completions.

    Internally delegates to ``run_async_loop`` with a
    ``BayesianOptimisationGenerator``.

    Parameters:
        total_evaluations:  Total number of simulation evaluations.
        batch_size:         Candidates generated per surrogate retraining.
        generate_acq_function: ``(surrogate, state) -> acquisition_function``.
        state:              Current ``State``.
        surrogate:          Surrogate model instance.
        simulator:          ``ExectuableSimulator`` or ``PythonSimulator``.
        resource_manager:   ``ResourceManager`` tracking available cores.
        scheduler:          ``Scheduler`` instance (required for
                            ``ExectuableSimulator``).
        fidelity_configs:   Optional ``{fidelity: FidelityConfig}`` mapping.
        retrain_interval:   Retrain surrogate every *N* completions
                            (default: *batch_size*).
        max_workers:        Thread-pool size (default 16).
        poll_interval:      Seconds between scheduling checks (default 0.5).
        csv_name:           Optional CSV path to persist state.
        verbose:            Enable info-level log messages.
        **kwargs:           Forwarded to ``suggest_next_locations``.

    Returns:
        Updated ``State``.
    """
    if isinstance(simulator, ExectuableSimulator) and scheduler is None:
        raise ValueError("If simulator is an ExectuableSimulator, you must provide a scheduler")
    assert isinstance(surrogate, BaseSurrogate)

    generator = BayesianOptimisationGenerator(
        domain=state.input_domain,
        surrogate=surrogate,
        generate_acq_fn=generate_acq_function,
        refit_surrogate=True,
        verbose=verbose,
        **kwargs,
    )

    return run_async_loop(
        total_evaluations=total_evaluations,
        generate_candidates=generator,
        state=state,
        simulator=simulator,
        resource_manager=resource_manager,
        scheduler=scheduler,
        fidelity_configs=fidelity_configs,
        refill_interval=retrain_interval or batch_size,
        batch_size=batch_size,
        max_workers=max_workers,
        poll_interval=poll_interval,
        csv_name=csv_name,
        verbose=verbose,
    )


def run_async_loop(
    total_evaluations,
    generate_candidates,
    state,
    simulator,
    resource_manager,
    scheduler=None,
    fidelity_configs=None,
    refill_interval=None,
    batch_size=None,
    max_workers=16,
    poll_interval=0.5,
    csv_name=None,
    verbose=False,
):
    """Generic asynchronous evaluation loop with a pluggable candidate generator.

    Accepts **any** ``CandidateGenerator`` instance (recommended) or a plain
    callable ``(state, budget) -> (Xs, Ss | None)``.

    Parameters:
        total_evaluations:
            Total number of simulation evaluations to perform.
        generate_candidates:
            Either a ``CandidateGenerator`` instance whose ``__call__``
            returns ``(indices, Xs, Ss)`` or a plain callable with
            signature ``(state, budget) -> (Xs, Ss | None)``.
        state:              Current ``State``.
        simulator:          ``ExectuableSimulator`` or ``PythonSimulator``.
        resource_manager:   ``ResourceManager`` tracking available cores.
        scheduler:          ``Scheduler`` instance (required for
                            ``ExectuableSimulator``).
        fidelity_configs:   Optional ``{fidelity: FidelityConfig}`` mapping.
        refill_interval:    Request new candidates every *N* completions
                            (default: *batch_size* or first batch size).
        batch_size:         Number of candidates per generation call.
                            Defaults to *refill_interval* or total cores.
        max_workers:        Thread-pool size (default 16).
        poll_interval:      Seconds between scheduling checks (default 0.5).
        csv_name:           Optional CSV path to persist state.
        verbose:            Enable info-level log messages.

    Returns:
        Updated ``State``.
    """
    if isinstance(simulator, ExectuableSimulator) and scheduler is None:
        raise ValueError("If simulator is an ExectuableSimulator, you must provide a scheduler")

    is_generator_cls = isinstance(generate_candidates, CandidateGenerator)

    async_sched = AsyncScheduler(
        simulator=simulator,
        resource_manager=resource_manager,
        scheduler=scheduler,
        fidelity_configs=fidelity_configs,
        max_workers=max_workers,
        poll_interval=poll_interval,
    )

    # --- initial candidates ------------------------------------------------
    initial_budget = batch_size or min(total_evaluations, resource_manager.total)
    initial_budget = min(initial_budget, total_evaluations)

    if is_generator_cls:
        idx_init, X_init, S_init = generate_candidates(state, initial_budget)
    else:
        result = generate_candidates(state, initial_budget)
        if isinstance(result, tuple) and len(result) == 3:
            idx_init, X_init, S_init = result
        elif isinstance(result, tuple) and len(result) == 2:
            X_init, S_init = result
            index_start = int(state.index.max()) + 1
            idx_init = index_start + np.arange(X_init.shape[0])
        else:
            X_init, S_init = result, None
            index_start = int(state.index.max()) + 1
            idx_init = index_start + np.arange(X_init.shape[0])

    n_init = X_init.shape[0]
    initial_tasks = async_sched.create_tasks(idx_init, X_init, S_init)

    if refill_interval is None:
        refill_interval = batch_size or n_init

    # --- book-keeping ------------------------------------------------------
    evaluations_launched = [n_init]
    completions_since_refill = [0]

    def _on_tasks_complete(state, completed_tasks):
        completions_since_refill[0] += len(completed_tasks)

        if csv_name is not None:
            state.to_csv(csv_name)

        remaining = total_evaluations - evaluations_launched[0]
        if remaining <= 0:
            return None

        if completions_since_refill[0] >= refill_interval:
            completions_since_refill[0] = 0
            budget = min(batch_size or refill_interval, remaining)

            if verbose:
                logger.info(
                    "Generating new candidates (launched=%d/%d)",
                    evaluations_launched[0],
                    total_evaluations,
                )

            if is_generator_cls:
                idx_new, X_new, S_new = generate_candidates(state, budget)
            else:
                result = generate_candidates(state, budget)
                if isinstance(result, tuple) and len(result) == 3:
                    idx_new, X_new, S_new = result
                elif isinstance(result, tuple) and len(result) == 2:
                    X_new, S_new = result
                    idx_start = int(state.index.max()) + 1
                    idx_new = idx_start + np.arange(X_new.shape[0])
                else:
                    X_new, S_new = result, None
                    idx_start = int(state.index.max()) + 1
                    idx_new = idx_start + np.arange(X_new.shape[0])

            n_new = X_new.shape[0]
            new_tasks = async_sched.create_tasks(idx_new, X_new, S_new)

            evaluations_launched[0] += n_new
            return new_tasks

        return None

    # --- run ---------------------------------------------------------------
    async_sched.run(state, initial_tasks, on_tasks_complete=_on_tasks_complete)

    return state
