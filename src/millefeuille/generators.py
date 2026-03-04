"""
Candidate generators for populating task queues.

Provides a generic interface for methods that generate candidate points
based on the current ``State``.  Each generator's ``__call__`` returns
``(indices, Xs, Ss)`` ready for consumption by the async scheduler or
a synchronous evaluation loop.

Key classes:
    CandidateGenerator                – abstract base class
    RandomCandidateGenerator          – uniform random sampling
    BayesianOptimisationGenerator     – surrogate + acquisition function
    ThresholdCandidateGenerator       – probabilistic threshold sampling
    ThresholdExclusionGenerator       – threshold + proximity exclusion
"""

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from .domain import InputDomain
from .optimise import suggest_next_locations
from .state import State
from .surrogate import BaseSurrogate


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class CandidateGenerator(ABC):
    """Abstract base class for candidate point generators.

    Subclasses implement ``generate`` which returns arrays of inputs and
    (optionally) fidelities.  The ``__call__`` method wraps ``generate``
    to also produce unique run indices derived from the current state.

    Parameters:
        domain: ``InputDomain`` defining the parameter space.
    """

    def __init__(self, domain: InputDomain):
        self.domain = domain

    @abstractmethod
    def generate(
        self,
        state: State,
        n_candidates: int,
    ) -> tuple[npt.NDArray, npt.NDArray | None]:
        """Generate candidate inputs.

        Parameters:
            state:        Current optimisation state.
            n_candidates: Number of candidates requested.

        Returns:
            Xs:  Input array of shape ``(N, dim)``.  ``N`` may be less
                 than *n_candidates* if the strategy cannot fill the budget.
            Ss:  Fidelity array of shape ``(N, 1)`` or ``None`` for
                 single-fidelity problems.
        """
        ...

    def __call__(
        self,
        state: State,
        n_candidates: int,
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray | None]:
        """Generate candidates and assign indices.

        Calls ``self.generate`` then builds contiguous indices starting
        from ``state.index.max() + 1``.

        Returns:
            indices:  1-D integer array of length ``N``.
            Xs:       Input array ``(N, dim)``.
            Ss:       Fidelity array ``(N, 1)`` or ``None``.
        """
        Xs, Ss = self.generate(state, n_candidates)
        n = Xs.shape[0]
        index_start = int(state.index.max()) + 1
        indices = index_start + np.arange(n)
        return indices, Xs, Ss


# ---------------------------------------------------------------------------
# Random sampling
# ---------------------------------------------------------------------------


class RandomCandidateGenerator(CandidateGenerator):
    """Uniform random sampling over the input domain.

    Parameters:
        domain:          ``InputDomain``.
        sampler:         QMC sampler instance with a ``.random(n)`` method
                         (e.g. ``scipy.stats.qmc.Sobol``).  If ``None`` a
                         plain uniform RNG is used.
        fidelity_probs:  Optional dict ``{fidelity_level: probability}`` for
                         random fidelity assignment, e.g. ``{0: 0.7, 1: 0.3}``.
                         ``None`` means single-fidelity.
        rng:             ``np.random.Generator`` (optional, used when
                         *sampler* is ``None``).
    """

    def __init__(
        self,
        domain: InputDomain,
        sampler=None,
        fidelity_probs: dict[int, float] | None = None,
        rng: np.random.Generator | None = None,
    ):
        super().__init__(domain)
        self.sampler = sampler
        self.fidelity_probs = fidelity_probs
        self._rng = rng or np.random.default_rng()

    def generate(self, state, n_candidates):
        if self.sampler is not None:
            X_unit = self.sampler.random(n_candidates)
        else:
            X_unit = self._rng.uniform(size=(n_candidates, self.domain.dim))

        Xs = self.domain.inverse_transform(X_unit)
        Ss = self._assign_fidelities(n_candidates)
        return Xs, Ss

    def _assign_fidelities(self, n: int) -> npt.NDArray | None:
        if self.fidelity_probs is None:
            return None
        fids = list(self.fidelity_probs.keys())
        probs = list(self.fidelity_probs.values())
        return self._rng.choice(fids, size=(n, 1), p=probs)


# ---------------------------------------------------------------------------
# Bayesian optimisation (surrogate + acquisition)
# ---------------------------------------------------------------------------


class BayesianOptimisationGenerator(CandidateGenerator):
    """Candidate generation via surrogate-based acquisition optimisation.

    Wraps ``suggest_next_locations`` with automatic surrogate retraining.

    Parameters:
        domain:          ``InputDomain``.
        surrogate:       Surrogate model instance.
        generate_acq_fn: Callable ``(surrogate, state) -> acq_function``.
        refit_surrogate: If ``True`` (default) the surrogate is refitted
                         on every call.
        **optim_kwargs:  Forwarded to ``suggest_next_locations`` (e.g.
                         ``num_restarts``, ``raw_samples``,
                         ``optimizer_options``, ``fixed_features``).
    """

    def __init__(
        self,
        domain: InputDomain,
        surrogate: BaseSurrogate,
        generate_acq_fn,
        refit_surrogate: bool = True,
        verbose: bool = False,
        **optim_kwargs,
    ):
        super().__init__(domain)
        self.surrogate = surrogate
        self.generate_acq_fn = generate_acq_fn
        self.refit_surrogate = refit_surrogate
        self.verbose = verbose
        self._optim_kwargs = optim_kwargs

    def generate(self, state, n_candidates):
        if self.refit_surrogate:
            self.surrogate.fit(state)

        acq_function = self.generate_acq_fn(self.surrogate, state)

        if state.l_MultiFidelity:
            Xs, Ss = suggest_next_locations(
                n_candidates,
                state,
                acq_function,
                verbose=self.verbose,
                **self._optim_kwargs,
            )
        else:
            Xs = suggest_next_locations(
                n_candidates,
                state,
                acq_function,
                verbose=self.verbose,
                **self._optim_kwargs,
            )
            Ss = None

        return Xs, Ss


# ---------------------------------------------------------------------------
# Probabilistic threshold sampling
# ---------------------------------------------------------------------------


class ThresholdCandidateGenerator(CandidateGenerator):
    """Surrogate-guided probabilistic threshold sampling.

    Draws a large pool of random candidates, predicts with the surrogate,
    and keeps those whose predicted probability of exceeding
    ``threshold_value`` is greater than a random draw.

    Parameters:
        domain:              ``InputDomain``.
        sampler:             QMC sampler with ``.random(n)`` method.
        surrogate:           Surrogate model instance.
        threshold_value:     Threshold for the probabilistic filter.
        pool_size:           Size of the initial random pool drawn each call.
        refit_surrogate:     Refit surrogate before predicting (default True).
        fidelity_probs:      Optional ``{fidelity: prob}`` for fidelity
                             assignment on selected candidates.
        target_fidelity:     Forwarded to surrogate prediction.
        target_key:          Forwarded to surrogate prediction.
    """

    def __init__(
        self,
        domain: InputDomain,
        sampler,
        surrogate: BaseSurrogate,
        threshold_value: float,
        pool_size: int = 256,
        refit_surrogate: bool = True,
        fidelity_probs: dict[int, float] | None = None,
        target_fidelity: int | None = None,
        target_key=None,
        rng: np.random.Generator | None = None,
    ):
        super().__init__(domain)
        self.sampler = sampler
        self.surrogate = surrogate
        self.threshold_value = threshold_value
        self.pool_size = pool_size
        self.refit_surrogate = refit_surrogate
        self.fidelity_probs = fidelity_probs
        self.target_fidelity = target_fidelity
        self.target_key = target_key
        self._rng = rng or np.random.default_rng()

    def generate(self, state, n_candidates):
        if self.refit_surrogate:
            self.surrogate.fit(state)

        x_all, _, _, mask = _probabilistic_threshold_filter(
            domain=self.domain,
            state=state,
            sampler=self.sampler,
            surrogate=self.surrogate,
            pool_size=self.pool_size,
            threshold_value=self.threshold_value,
            target_fidelity=self.target_fidelity,
            target_key=self.target_key,
        )

        x_pass = x_all[mask]
        if len(x_pass) == 0:
            # Fallback: uniform random
            x_pass = self.domain.inverse_transform(
                self._rng.uniform(size=(n_candidates, self.domain.dim))
            )
        elif len(x_pass) > n_candidates:
            idx = self._rng.choice(len(x_pass), size=n_candidates, replace=False)
            x_pass = x_pass[idx]

        Ss = self._assign_fidelities(len(x_pass))
        return x_pass, Ss

    def _assign_fidelities(self, n: int) -> npt.NDArray | None:
        if self.fidelity_probs is None:
            return None
        fids = list(self.fidelity_probs.keys())
        probs = list(self.fidelity_probs.values())
        return self._rng.choice(fids, size=(n, 1), p=probs)


# ---------------------------------------------------------------------------
# Threshold sampling with exclusion
# ---------------------------------------------------------------------------


class ThresholdExclusionGenerator(CandidateGenerator):
    """Probabilistic threshold sampling with proximity exclusion.

    Like ``ThresholdCandidateGenerator`` but applies PCA-normalised
    distance-based rejection to avoid tightly clustered candidates.

    Parameters:
        domain:              ``InputDomain``.
        sampler:             QMC sampler with ``.random(n)`` method.
        surrogate:           Surrogate model instance.
        threshold_value:     Threshold for the probabilistic filter.
        rejection_radius:    Minimum distance in PCA-normalised space.
        pool_size:           Size of the initial random pool drawn each call.
        n_clusters:          Number of clusters for PCA analysis (default 1).
        refit_surrogate:     Refit surrogate before predicting (default True).
        fidelity_probs:      Optional ``{fidelity: prob}`` for fidelity
                             assignment on selected candidates.
        target_fidelity:     Forwarded to surrogate prediction.
        target_key:          Forwarded to surrogate prediction.
    """

    def __init__(
        self,
        domain: InputDomain,
        sampler,
        surrogate: BaseSurrogate,
        threshold_value: float,
        rejection_radius: float,
        pool_size: int = 256,
        n_clusters: int = 1,
        refit_surrogate: bool = True,
        fidelity_probs: dict[int, float] | None = None,
        target_fidelity: int | None = None,
        target_key=None,
        rng: np.random.Generator | None = None,
    ):
        super().__init__(domain)
        self.sampler = sampler
        self.surrogate = surrogate
        self.threshold_value = threshold_value
        self.rejection_radius = rejection_radius
        self.pool_size = pool_size
        self.n_clusters = n_clusters
        self.refit_surrogate = refit_surrogate
        self.fidelity_probs = fidelity_probs
        self.target_fidelity = target_fidelity
        self.target_key = target_key
        self._rng = rng or np.random.default_rng()

    def generate(self, state, n_candidates):
        if self.refit_surrogate:
            self.surrogate.fit(state)

        x_all, y_pred, prob, mask = _probabilistic_threshold_filter(
            domain=self.domain,
            state=state,
            sampler=self.sampler,
            surrogate=self.surrogate,
            pool_size=self.pool_size,
            threshold_value=self.threshold_value,
            target_fidelity=self.target_fidelity,
            target_key=self.target_key,
        )

        x_candidates = x_all[mask]
        prob_candidates = prob[mask]

        if len(x_candidates) == 0:
            # Fallback: uniform random
            Xs = self.domain.inverse_transform(
                self._rng.uniform(size=(n_candidates, self.domain.dim))
            )
            Ss = self._assign_fidelities(n_candidates)
            return Xs, Ss

        # Apply proximity exclusion
        x_selected, _, _ = _greedy_exclusion(
            x_candidates, y_pred[mask], prob_candidates,
            n_candidates, self.rejection_radius, self.n_clusters,
        )

        if len(x_selected) == 0:
            Xs = self.domain.inverse_transform(
                self._rng.uniform(size=(n_candidates, self.domain.dim))
            )
        else:
            Xs = x_selected

        Ss = self._assign_fidelities(len(Xs))
        return Xs, Ss

    def _assign_fidelities(self, n: int) -> npt.NDArray | None:
        if self.fidelity_probs is None:
            return None
        fids = list(self.fidelity_probs.keys())
        probs = list(self.fidelity_probs.values())
        return self._rng.choice(fids, size=(n, 1), p=probs)


# ---------------------------------------------------------------------------
# Shared helper functions (private)
# ---------------------------------------------------------------------------


def _probabilistic_threshold_filter(
    domain,
    state,
    sampler,
    surrogate,
    pool_size,
    threshold_value,
    target_fidelity=None,
    target_key=None,
    random_draws=None,
):
    """Draw random candidates and compute probabilistic threshold mask.

    Returns ``(x_all, y_pred, prob, mask)`` — same contract as the
    top-level ``probabilistic_threshold_sampling`` in ``utils.py``.
    """
    x_unit = sampler.random(pool_size)
    x_all = domain.inverse_transform(x_unit)

    predictions = surrogate.predict(state, x_all)

    if target_key is not None:
        prediction = predictions[target_key]
    elif target_fidelity is not None:
        prediction = predictions[target_fidelity]
    else:
        if "mean" not in predictions.keys():
            raise ValueError(
                "mean missing from predictions.keys(), did you miss target_fidelity or target_key inputs?"
            )
        prediction = predictions

    mean = prediction["mean"].flatten()
    std = prediction["std"].flatten()
    prob = 1.0 - norm.cdf(threshold_value, loc=mean, scale=std)

    if random_draws is None:
        random_draws = np.random.rand(pool_size)
    mask = prob > random_draws

    return x_all, mean, prob, mask


def _greedy_exclusion(
    x_candidates,
    y_candidates,
    prob_candidates,
    batch_size,
    rejection_radius,
    n_clusters=1,
):
    """Greedy selection with PCA-normalised proximity exclusion.

    Extracted from ``probabilistic_threshold_sampling_with_exclusion``
    in ``utils.py`` so it can be shared.
    """
    n_candidates, n_dims = x_candidates.shape

    n_clusters_actual = min(n_clusters, n_candidates)
    if n_clusters_actual > 1:
        from scipy.cluster.vq import kmeans2

        _, labels = kmeans2(x_candidates, n_clusters_actual, minit="points", seed=0)
    else:
        labels = np.zeros(n_candidates, dtype=int)
        n_clusters_actual = 1

    # PCA-normalise within each cluster
    x_normalised = np.empty_like(x_candidates)
    for k in range(n_clusters_actual):
        cluster_mask = labels == k
        x_cluster = x_candidates[cluster_mask]

        if len(x_cluster) == 0:
            continue

        mean = x_cluster.mean(axis=0)
        x_centered = x_cluster - mean

        if len(x_cluster) == 1 or n_dims == 1:
            std = x_centered.std(axis=0)
            std = np.where(std > 0, std, 1.0)
            x_normalised[cluster_mask] = x_centered / std
        else:
            cov = np.cov(x_centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            eigenvalues = np.maximum(eigenvalues, 1e-10)
            x_normalised[cluster_mask] = x_centered @ eigenvectors / np.sqrt(eigenvalues)

    # Greedy selection, highest probability first
    sort_idx = np.argsort(-prob_candidates)

    selected_indices = []
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
    return (
        x_candidates[selected_indices],
        y_candidates[selected_indices],
        prob_candidates[selected_indices],
    )
