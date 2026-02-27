import numpy as np
from scipy.stats import norm

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

    # Step 4: Greedy selection with rejection based on distance in normalised space
    # Prioritise highest-probability candidates
    sort_idx = np.argsort(-prob_candidates)

    selected_indices = []
    selected_normalised = []

    for idx in sort_idx:
        if len(selected_indices) >= batch_size:
            break

        x_norm_i = x_normalised[idx]

        if len(selected_normalised) == 0:
            selected_indices.append(idx)
            selected_normalised.append(x_norm_i)
        else:
            dists = np.linalg.norm(np.array(selected_normalised) - x_norm_i, axis=1)
            if np.all(dists >= rejection_radius):
                selected_indices.append(idx)
                selected_normalised.append(x_norm_i)

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
