"""
Candidate generators for populating task queues.

Provides a composable interface for methods that generate candidate points
based on the current ``State``.  Each generator's ``__call__`` returns
``(indices, Xs, Ss)`` ready for consumption by the async scheduler or
a synchronous evaluation loop.

**Design philosophy — composable generators**

Generators fall into two categories:

*Base generators* produce candidates from scratch:
    ``RandomCandidateGenerator``              — uniform / QMC sampling
    ``BayesianOptimisationGenerator``         — surrogate + acquisition
    ``ThresholdCandidateGenerator``           — probabilistic threshold sampling
    ``SurrogateThresholdCandidateGenerator``  — deterministic surrogate threshold
    ``MetropolisHastingsGenerator``           — MCMC chain sampling via surrogate posterior

*Wrapper generators* refine the output of another generator:
    ``GreedyExclusionGenerator``             — proximity-based exclusion

Because every generator implements the same ``CandidateGenerator`` interface
they can be freely composed::

    generator = GreedyExclusionGenerator(
        inner=ThresholdCandidateGenerator(domain, sampler, surrogate, ...),
        rejection_radius=0.5,
    )
    indices, Xs, Ss = generator(state, batch_size)

Public helper functions:
    ``probabilistic_threshold_filter``  — draw + predict + threshold mask
    ``greedy_exclusion``                — PCA-normalised proximity selection
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
        index_start = int(state.index.max()) + 1 if state.index is not None else 0
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

    If fewer than ``n_candidates`` pass the filter on the first attempt
    the pool size is multiplied by ``pool_try_multiplier`` and the draw
    is repeated, up to ``max_retries`` times.  A ``RuntimeError`` is
    raised if insufficient candidates are found after all retries.

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
        min_probability:     Minimum probability for a candidate to pass
                             the stochastic filter (default 0.0).  Random
                             draws are sampled from
                             ``[min_probability, 1]`` instead of ``[0, 1]``.
        sort_candidates:     If ``True`` the returned candidates are sorted
                             by descending probability (default ``False``).
        max_retries:         Maximum number of pool-expansion retries
                             (default 5).
        pool_try_multiplier: Factor by which the pool size grows on each
                             retry (default 2).
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
        min_probability: float = 0.0,
        sort_candidates: bool = False,
        max_retries: int = 5,
        pool_try_multiplier: int = 2,
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
        self.min_probability = min_probability
        self.sort_candidates = sort_candidates
        self.max_retries = max_retries
        self.pool_try_multiplier = pool_try_multiplier
        self._rng = rng or np.random.default_rng()

    def generate(self, state, n_candidates):
        if self.refit_surrogate:
            self.surrogate.fit(state)

        current_pool = self.pool_size
        for attempt in range(1 + self.max_retries):
            x_all, _, prob, mask = probabilistic_threshold_filter(
                domain=self.domain,
                state=state,
                sampler=self.sampler,
                surrogate=self.surrogate,
                pool_size=current_pool,
                threshold_value=self.threshold_value,
                target_fidelity=self.target_fidelity,
                target_key=self.target_key,
                min_probability=self.min_probability,
            )

            x_pass = x_all[mask]
            prob_pass = prob[mask]

            if len(x_pass) >= n_candidates:
                break
            current_pool *= self.pool_try_multiplier

        if len(x_pass) == 0:
            raise RuntimeError(
                f"ThresholdCandidateGenerator: no candidates passed the "
                f"threshold after {1 + self.max_retries} attempts "
                f"(final pool size {current_pool})."
            )
        if len(x_pass) < n_candidates:
            raise RuntimeError(
                f"ThresholdCandidateGenerator: only {len(x_pass)} of "
                f"{n_candidates} requested candidates passed the threshold "
                f"after {1 + self.max_retries} attempts "
                f"(final pool size {current_pool})."
            )

        if self.sort_candidates:
            order = np.argsort(-prob_pass)
            x_pass = x_pass[order]

        if len(x_pass) > n_candidates:
            x_pass = x_pass[:n_candidates]

        Ss = self._assign_fidelities(len(x_pass))
        return x_pass, Ss

    def _assign_fidelities(self, n: int) -> npt.NDArray | None:
        if self.fidelity_probs is None:
            return None
        fids = list(self.fidelity_probs.keys())
        probs = list(self.fidelity_probs.values())
        return self._rng.choice(fids, size=(n, 1), p=probs)


# ---------------------------------------------------------------------------
# Deterministic surrogate threshold sampling
# ---------------------------------------------------------------------------


class SurrogateThresholdCandidateGenerator(CandidateGenerator):
    """Deterministic surrogate threshold sampling.

    Draws a pool of random candidates, evaluates the surrogate, and keeps
    those whose predicted mean exceeds ``threshold_value``.

    If fewer than ``n_candidates`` pass the filter on the first attempt
    the pool size is multiplied by ``pool_try_multiplier`` and the draw
    is repeated, up to ``max_retries`` times.  A ``RuntimeError`` is
    raised if insufficient candidates are found after all retries.

    Parameters:
        domain:              ``InputDomain``.
        sampler:             QMC sampler with ``.random(n)`` method.
        surrogate:           Surrogate model instance.
        threshold_value:     Deterministic threshold on predicted mean.
        pool_size:           Size of the initial random pool drawn each call.
        refit_surrogate:     Refit surrogate before predicting (default True).
        fidelity_probs:      Optional ``{fidelity: prob}`` for fidelity
                             assignment on selected candidates.
        target_fidelity:     Forwarded to surrogate prediction.
        target_key:          Forwarded to surrogate prediction.
        sort_candidates:     If ``True`` the returned candidates are sorted
                             by descending predicted mean (default ``False``).
        max_retries:         Maximum number of pool-expansion retries
                             (default 5).
        pool_try_multiplier: Factor by which the pool size grows on each
                             retry (default 2).
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
        sort_candidates: bool = False,
        max_retries: int = 5,
        pool_try_multiplier: int = 2,
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
        self.sort_candidates = sort_candidates
        self.max_retries = max_retries
        self.pool_try_multiplier = pool_try_multiplier
        self._rng = rng or np.random.default_rng()

    def generate(self, state, n_candidates):
        if self.refit_surrogate:
            self.surrogate.fit(state)

        current_pool = self.pool_size
        for attempt in range(1 + self.max_retries):
            x_unit = self.sampler.random(current_pool)
            x_all = self.domain.inverse_transform(x_unit)

            predictions = self.surrogate.predict(state, x_all)

            if self.target_key is not None:
                prediction = predictions[self.target_key]
            elif self.target_fidelity is not None:
                prediction = predictions[self.target_fidelity]
            else:
                if "mean" not in predictions.keys():
                    raise ValueError(
                        "mean missing from predictions.keys(), did you miss target_fidelity or target_key inputs?"
                    )
                prediction = predictions

            mean = prediction["mean"].flatten()
            mask = mean > self.threshold_value

            x_pass = x_all[mask]
            mean_pass = mean[mask]

            if len(x_pass) >= n_candidates:
                break
            current_pool *= self.pool_try_multiplier

        if len(x_pass) == 0:
            raise RuntimeError(
                f"SurrogateThresholdCandidateGenerator: no candidates exceeded "
                f"the threshold after {1 + self.max_retries} attempts "
                f"(final pool size {current_pool})."
            )
        if len(x_pass) < n_candidates:
            raise RuntimeError(
                f"SurrogateThresholdCandidateGenerator: only {len(x_pass)} of "
                f"{n_candidates} requested candidates exceeded the threshold "
                f"after {1 + self.max_retries} attempts "
                f"(final pool size {current_pool})."
            )

        if self.sort_candidates:
            order = np.argsort(-mean_pass)
            x_pass = x_pass[order]

        if len(x_pass) > n_candidates:
            x_pass = x_pass[:n_candidates]

        Ss = self._assign_fidelities(len(x_pass))
        return x_pass, Ss

    def _assign_fidelities(self, n: int) -> npt.NDArray | None:
        if self.fidelity_probs is None:
            return None
        fids = list(self.fidelity_probs.keys())
        probs = list(self.fidelity_probs.values())
        return self._rng.choice(fids, size=(n, 1), p=probs)


# ---------------------------------------------------------------------------
# Greedy exclusion wrapper (composable)
# ---------------------------------------------------------------------------


class GreedyExclusionGenerator(CandidateGenerator):
    """Composable wrapper that applies proximity-based exclusion.

    Requests a larger pool of candidates from an inner generator, then
    greedily selects well-spaced points using PCA-normalised distance.
    Candidates are considered in the order returned by the inner generator,
    so the inner generator controls priority (e.g.
    ``ThresholdCandidateGenerator`` sorts by descending probability).

    Compose with any base generator::

        generator = GreedyExclusionGenerator(
            inner=ThresholdCandidateGenerator(domain, sampler, surrogate, ...),
            rejection_radius=0.5,
        )

    Parameters:
        inner:               Another ``CandidateGenerator`` to draw from.
        rejection_radius:    Minimum distance in PCA-normalised space.
        pool_multiplier:     How many multiples of ``n_candidates`` to
                             request from *inner* (default 4).
        n_clusters:          Number of clusters for PCA analysis (default 1).
    """

    def __init__(
        self,
        inner: CandidateGenerator,
        rejection_radius: float,
        pool_multiplier: int = 4,
        n_clusters: int = 1,
    ):
        super().__init__(inner.domain)
        self.inner = inner
        self.rejection_radius = rejection_radius
        self.pool_multiplier = pool_multiplier
        self.n_clusters = n_clusters

    def generate(self, state, n_candidates):
        pool_request = n_candidates * self.pool_multiplier
        Xs, Ss = self.inner.generate(state, pool_request)

        if len(Xs) == 0:
            return Xs, Ss

        if len(Xs) <= n_candidates:
            return Xs, Ss

        selected_idx = greedy_exclusion(Xs, n_candidates, self.rejection_radius, self.n_clusters)

        if len(selected_idx) == 0:
            return Xs[:n_candidates], Ss[:n_candidates] if Ss is not None else None

        Xs_out = Xs[selected_idx]
        Ss_out = Ss[selected_idx] if Ss is not None else None
        return Xs_out, Ss_out


# ---------------------------------------------------------------------------
# Threshold sampling with exclusion (convenience)
# ---------------------------------------------------------------------------


class ThresholdExclusionGenerator(CandidateGenerator):
    """Probabilistic threshold sampling with proximity exclusion.

    Equivalent to composing ``ThresholdCandidateGenerator`` inside
    ``GreedyExclusionGenerator`` — provided as a convenience class.

    Like ``ThresholdCandidateGenerator`` but applies PCA-normalised
    distance-based rejection to avoid tightly clustered candidates.

    If fewer than ``n_candidates`` survive the threshold *and* exclusion
    steps, the pool size is multiplied by ``pool_try_multiplier`` and
    the draw is repeated, up to ``max_retries`` times.  A
    ``RuntimeError`` is raised if insufficient candidates are found.

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
        sort_candidates:     If ``True`` candidates passed to the exclusion
                             step are sorted by descending probability
                             (default ``True``).  The exclusion step
                             always processes candidates in the order
                             given.
        max_retries:         Maximum number of pool-expansion retries
                             (default 5).
        pool_try_multiplier: Factor by which the pool size grows on each
                             retry (default 2).
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
        sort_candidates: bool = True,
        max_retries: int = 5,
        pool_try_multiplier: int = 2,
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
        self.sort_candidates = sort_candidates
        self.max_retries = max_retries
        self.pool_try_multiplier = pool_try_multiplier
        self._rng = rng or np.random.default_rng()

    def generate(self, state, n_candidates):
        if self.refit_surrogate:
            self.surrogate.fit(state)

        current_pool = self.pool_size
        for attempt in range(1 + self.max_retries):
            x_all, y_pred, prob, mask = probabilistic_threshold_filter(
                domain=self.domain,
                state=state,
                sampler=self.sampler,
                surrogate=self.surrogate,
                pool_size=current_pool,
                threshold_value=self.threshold_value,
                target_fidelity=self.target_fidelity,
                target_key=self.target_key,
            )

            x_candidates = x_all[mask]
            prob_candidates = prob[mask]

            if len(x_candidates) == 0:
                current_pool *= self.pool_try_multiplier
                continue

            if self.sort_candidates:
                order = np.argsort(-prob_candidates)
                x_candidates = x_candidates[order]

            selected_idx = greedy_exclusion(x_candidates, n_candidates, self.rejection_radius, self.n_clusters)

            if len(selected_idx) >= n_candidates:
                Xs = x_candidates[selected_idx]
                Ss = self._assign_fidelities(len(Xs))
                return Xs, Ss

            current_pool *= self.pool_try_multiplier

        # All retries exhausted
        n_survived = len(selected_idx) if len(x_candidates) > 0 else 0
        raise RuntimeError(
            f"ThresholdExclusionGenerator: only {n_survived} of "
            f"{n_candidates} requested candidates survived threshold + "
            f"exclusion after {1 + self.max_retries} attempts "
            f"(final pool size {current_pool // self.pool_try_multiplier})."
        )

    def _assign_fidelities(self, n: int) -> npt.NDArray | None:
        if self.fidelity_probs is None:
            return None
        fids = list(self.fidelity_probs.keys())
        probs = list(self.fidelity_probs.values())
        return self._rng.choice(fids, size=(n, 1), p=probs)


# ---------------------------------------------------------------------------
# Metropolis-Hastings MCMC candidate generator
# ---------------------------------------------------------------------------


class MetropolisHastingsGenerator(CandidateGenerator):
    """MCMC candidate generator using a Metropolis-Hastings random walk.

    Runs ``n_chains`` independent Markov chains, each seeded from the last
    ``n_chains`` observed points in *state* (sorted by observation order).
    Proposals are Gaussian random walks in the **unit-cube** (transformed)
    space, clamped to ``[0, 1]^d`` before back-transforming.

    The acceptance test is Thompson-sampling style: for both the current
    and proposed point a scalar score is drawn from
    ``N(mean(x), std(x)^2)`` using the surrogate posterior.  The proposal
    is accepted with probability

    .. math::

        \\alpha = \\min\\!\\left(1,\\; \\exp(f' - f)\\right)

    so the chain drifts towards regions of higher predicted value.

    N.B. Ys must represent log likelihoods or similar, i.e. higher is
    better.

    After ``n_burnin`` warm-up steps, the chains collect samples.  The
    total pool across all chains is randomly thinned to return exactly
    ``n_candidates`` points.

    Parameters:
        domain:          ``InputDomain``.
        surrogate:       Surrogate model instance.
        proposal_std:    Standard deviation of the Gaussian proposal in
                         unit-cube space (default ``0.05``).
        n_burnin:        Number of burn-in steps discarded per chain
                         (default ``0``).
        n_steps:         Number of post-burn-in steps collected per chain
                         (default ``20``).  Total pool size is
                         ``n_chains * n_steps``; must be ``>= n_candidates``.
        n_chains:        Number of independent chains (default ``4``).
        refit_surrogate: Refit surrogate before running chains (default
                         ``True``).
        target_fidelity: Selects a fidelity key from the surrogate
                         ``predict`` output dict (optional).
        target_key:      Selects an output key from the surrogate
                         ``predict`` output dict (optional).
        rng:             ``np.random.Generator`` (optional).
    """

    def __init__(
        self,
        domain: InputDomain,
        surrogate: BaseSurrogate,
        proposal_std: float = 0.05,
        n_burnin: int = 0,
        n_steps: int = 20,
        n_chains: int = 4,
        refit_surrogate: bool = True,
        target_fidelity: int | None = None,
        target_key=None,
        rng: np.random.Generator | None = None,
    ):
        super().__init__(domain)
        self.surrogate = surrogate
        self.proposal_std = proposal_std
        self.n_burnin = n_burnin
        self.n_steps = n_steps
        self.n_chains = n_chains
        self.refit_surrogate = refit_surrogate
        self.target_fidelity = target_fidelity
        self.target_key = target_key
        self._rng = rng or np.random.default_rng()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_batch(self, state: State, x_raw: npt.NDArray) -> npt.NDArray:
        """Draw Thompson sample scores for a batch of points.

        Parameters:
            state: Current optimisation state.
            x_raw: Raw (un-transformed) inputs of shape ``(N, dim)``.

        Returns:
            Float array of shape ``(N,)`` drawn from
            ``N(mean(x_i), std(x_i)^2)`` independently per point.
        """
        predictions = self.surrogate.predict(state, x_raw)

        if self.target_key is not None:
            pred = predictions[self.target_key]
        elif self.target_fidelity is not None:
            pred = predictions[self.target_fidelity]
        else:
            pred = predictions

        mean = pred["mean"].flatten()
        std = np.maximum(pred["std"].flatten(), 0.0)
        return self._rng.normal(mean, std)

    def _seed_points_unit(self, state: State) -> npt.NDArray:
        """Return ``n_chains`` seed points in unit-cube space.

        Uses the last ``n_chains`` observed inputs from *state*
        (by position in the array, i.e. most recently appended).
        Falls back to uniform random if the state has fewer points
        than ``n_chains``.

        Returns:
            Array of shape ``(n_chains, dim)``.
        """
        if state.Xs is not None and len(state.Xs) >= self.n_chains:
            seeds_raw = state.Xs[-self.n_chains :]
        elif state.Xs is not None and len(state.Xs) > 0:
            # Fewer observed points than chains — tile up to n_chains
            repeats = int(np.ceil(self.n_chains / len(state.Xs)))
            seeds_raw = np.tile(state.Xs, (repeats, 1))[: self.n_chains]
        else:
            # No observations yet — uniform random seeds
            seeds_raw = self.domain.inverse_transform(self._rng.uniform(size=(self.n_chains, self.domain.dim)))

        # Transform to unit-cube space for proposals
        return state.transform_X(seeds_raw)

    # ------------------------------------------------------------------
    # CandidateGenerator interface
    # ------------------------------------------------------------------

    def generate(self, state: State, n_candidates: int) -> tuple[npt.NDArray, None]:
        if state.l_MultiFidelity:
            raise NotImplementedError("MetropolisHastingsGenerator does not currently support multi-fidelity problems.")

        pool_size = self.n_chains * self.n_steps
        if pool_size < n_candidates:
            raise ValueError(
                f"MetropolisHastingsGenerator: pool size (n_chains={self.n_chains} * "
                f"n_steps={self.n_steps} = {pool_size}) must be >= "
                f"n_candidates={n_candidates}."
            )

        if self.refit_surrogate:
            self.surrogate.fit(state)

        # x_unit / x_raw carry the state of all chains simultaneously
        x_unit = self._seed_points_unit(state)  # (n_chains, dim)
        x_raw = self.domain.inverse_transform(x_unit)  # (n_chains, dim)
        f_current = self._score_batch(state, x_raw)  # (n_chains,)

        collected: list[npt.NDArray] = []
        total_steps = self.n_burnin + self.n_steps

        for step in range(total_steps):
            # Propose for all chains, rejecting out-of-bounds proposals
            noise = self._rng.normal(0.0, self.proposal_std, size=(self.n_chains, self.domain.dim))
            x_unit_proposed = x_unit + noise
            # Out of bounds check
            out_of_bounds = np.any(x_unit_proposed < 0.0, axis=1) | np.any(x_unit_proposed > 1.0, axis=1)  # (n_chains,)
            if np.all(out_of_bounds):
                # All proposals out of bounds — skip this step
                continue

            # Only compute the following for in-bounds proposals; keep the rest as the current point
            x_unit_proposed[out_of_bounds] = x_unit[out_of_bounds]
            x_raw_proposed = self.domain.inverse_transform(x_unit_proposed)

            # Score all proposals in a single batched surrogate call
            f_proposed = np.where(out_of_bounds, -np.inf, self._score_batch(state, x_raw_proposed))  # (n_chains,)

            # Vectorised MH acceptance
            log_alpha = f_proposed - f_current
            accept = (log_alpha >= 0.0) | (
                self._rng.uniform(size=self.n_chains) < np.clip(np.exp(log_alpha), None, 1.0)
            )  # (n_chains,)

            mask = accept[:, np.newaxis]
            x_unit = np.where(mask, x_unit_proposed, x_unit)
            x_raw = np.where(mask, x_raw_proposed, x_raw)
            f_current = np.where(accept, f_proposed, f_current)

            if step >= self.n_burnin:
                collected.append(x_raw.copy())  # each entry: (n_chains, dim)

        # Stack collected steps → (n_steps * n_chains, dim), then thin
        pool = np.concatenate(collected, axis=0)
        chosen = self._rng.choice(len(pool), size=n_candidates, replace=False)
        return pool[chosen], None


# ---------------------------------------------------------------------------
# Public helper functions
# ---------------------------------------------------------------------------


def probabilistic_threshold_filter(
    domain,
    state,
    sampler,
    surrogate,
    pool_size,
    threshold_value,
    target_fidelity=None,
    target_key=None,
    random_draws=None,
    min_probability=0.0,
):
    """Draw random candidates and compute probabilistic threshold mask.

    Parameters:
        domain:          ``InputDomain`` object.
        state:           ``State`` object.
        sampler:         QMC sampler with ``.random(n)`` method.
        surrogate:       Surrogate model (must support ``predict(state, X)``).
        pool_size:       Number of random candidates to draw.
        threshold_value: Threshold value to compare against.
        target_fidelity: int — selects surrogate fidelity.
        target_key:      int or str — selects surrogate output.
        random_draws:    Optional ``np.ndarray`` of shape ``(pool_size,)``
                         with values in ``[0, 1]``.  If ``None``, draws are
                         generated from ``[min_probability, 1]``.
        min_probability: Lower bound for random draws (default 0.0).

    Returns:
        x_all:  All sampled input points ``(pool_size, dim)``.
        y_pred: Predicted means ``(pool_size,)``.
        prob:   Predicted ``P(y > threshold)`` for each point.
        mask:   Boolean array of points that passed the stochastic filter.
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
            raise ValueError("mean missing from predictions.keys(), did you miss target_fidelity or target_key inputs?")
        prediction = predictions

    mean = prediction["mean"].flatten()
    std = prediction["std"].flatten()
    prob = 1.0 - norm.cdf(threshold_value, loc=mean, scale=std)

    if random_draws is None:
        random_draws = min_probability + (1.0 - min_probability) * np.random.rand(pool_size)
    mask = prob > random_draws

    return x_all, mean, prob, mask


def greedy_exclusion(
    x_candidates,
    batch_size,
    rejection_radius,
    n_clusters=1,
):
    """Greedy selection with PCA-normalised proximity exclusion.

    Candidates are considered in the order given — the caller determines
    priority.  Returns the *indices* into ``x_candidates`` of the
    selected points.

    Parameters:
        x_candidates:     Input array ``(N, dim)``.
        batch_size:       Maximum number of points to select.
        rejection_radius: Minimum distance in PCA-normalised space.
        n_clusters:       Number of clusters for PCA analysis (default 1).

    Returns:
        selected_indices: 1-D integer array of indices into *x_candidates*.
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

    # Greedy selection in given order
    selected_indices = []
    selected_normalised_per_cluster = {k: [] for k in range(n_clusters_actual)}

    for idx in range(n_candidates):
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

    return np.array(selected_indices, dtype=int) if selected_indices else np.empty(0, dtype=int)
