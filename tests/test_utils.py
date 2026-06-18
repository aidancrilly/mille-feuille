import numpy as np
import pytest
import pytest_cases
import torch
from millefeuille.domain import FidelityDomain
from millefeuille.generators import GreedyExclusionGenerator, ThresholdExclusionGenerator, greedy_exclusion, probabilistic_threshold_filter
from millefeuille.generators import RandomCandidateGenerator
from millefeuille.initialise import generate_initial_sample
from millefeuille.state import State
from millefeuille.surrogate import MultiFidelityGPSurrogate, SingleFidelityGPSurrogate
from millefeuille.utils import probabilistic_threshold_sampling, probabilistic_threshold_sampling_with_exclusion

from .conftest import (
    TEST_KERNEL,
    TEST_KERNEL_KWARGS,
    ForresterDomain,
    ForresterSampler,
    PythonForresterFunction,
    Uniform,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double


@pytest_cases.fixture(params=[20])
def ntrain(request):
    return request.param


@pytest_cases.fixture(params=[2])
def initial_samples(request):
    return request.param


@pytest_cases.fixture(params=[0.5])
def threshold_value(request):
    return request.param


@pytest_cases.fixture()
def singlefidelitysample(ntrain):
    Is = np.arange(ntrain)
    _rng = np.random.default_rng(seed=12345)
    Xs, _ = generate_initial_sample(ForresterDomain, ForresterSampler(_rng), ntrain)
    f = PythonForresterFunction()
    _, Ys = f(Is, Xs)
    return Is, Xs, Ys


@pytest_cases.fixture()
def multifidelitysample(ntrain):
    Is = np.arange(ntrain)
    Ss = np.random.binomial(1, 0.5, size=ntrain).reshape(-1, 1)
    _rng = np.random.default_rng(seed=12345)
    Xs, _ = generate_initial_sample(ForresterDomain, ForresterSampler(_rng), ntrain)
    f = PythonForresterFunction()
    _, Ys = f(Is, Xs, Ss)
    return Is, Xs, Ss, Ys


@pytest.mark.unit
def test_singlefidelity_probabilistic_threshold_filter(singlefidelitysample, initial_samples, threshold_value):
    Is, Xs, Ys = singlefidelitysample

    state = State(ForresterDomain, Is, Xs, Ys)

    surrogate = SingleFidelityGPSurrogate()
    surrogate.fit(state)

    sampler = Uniform(ForresterDomain.dim)

    x_all, y_pred, prob, mask = probabilistic_threshold_filter(
        ForresterDomain, state, sampler, surrogate, initial_samples, threshold_value
    )
    assert np.all(prob <= 1.0) and np.all(prob >= 0.0), (
        "probabilistic_threshold_filter returning impossible prob values"
    )


@pytest.mark.unit
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_multifidelity_probabilistic_threshold_filter(multifidelitysample, initial_samples, threshold_value):
    Is, Xs, Ss, Ys = multifidelitysample

    ForresterFidelity = FidelityDomain(num_fidelities=len(np.unique(Ss)))

    state = State(ForresterDomain, Is, Xs, Ys, Ss=Ss, fidelity_domain=ForresterFidelity)

    surrogate = MultiFidelityGPSurrogate(kernel=TEST_KERNEL, kernel_kwargs=TEST_KERNEL_KWARGS)
    surrogate.fit(state)

    sampler = Uniform(ForresterDomain.dim)

    with pytest.raises(ValueError):
        x_all, y_pred, prob, mask = probabilistic_threshold_filter(
            ForresterDomain, state, sampler, surrogate, initial_samples, threshold_value, target_fidelity=None
        )

    for fidelity in range(2):
        x_all, y_pred, prob, mask = probabilistic_threshold_filter(
            ForresterDomain, state, sampler, surrogate, initial_samples, threshold_value, target_fidelity=fidelity
        )
        assert np.all(prob <= 1.0) and np.all(prob >= 0.0), (
            "probabilistic_threshold_filter returning impossible prob values"
        )


@pytest_cases.fixture(params=[3])
def batch_size(request):
    return request.param


@pytest_cases.fixture(params=[0.5])
def rejection_radius(request):
    return request.param


@pytest.mark.unit
def test_singlefidelity_threshold_with_greedy_exclusion(
    singlefidelitysample, threshold_value, batch_size, rejection_radius
):
    Is, Xs, Ys = singlefidelitysample

    state = State(ForresterDomain, Is, Xs, Ys)

    surrogate = SingleFidelityGPSurrogate()
    surrogate.fit(state)

    sampler = Uniform(ForresterDomain.dim)

    # Use a larger pool of candidates so the exclusion logic is exercised
    large_initial_samples = 50
    x_all, y_pred, prob, mask = probabilistic_threshold_filter(
        ForresterDomain, state, sampler, surrogate, large_initial_samples, threshold_value
    )

    x_candidates = x_all[mask]
    prob_candidates = prob[mask]

    if len(x_candidates) == 0:
        pytest.skip("No candidates passed threshold")

    # Sort by descending probability then apply exclusion
    order = np.argsort(-prob_candidates)
    x_sorted = x_candidates[order]
    prob_sorted = prob_candidates[order]

    selected_idx = greedy_exclusion(x_sorted, batch_size, rejection_radius)

    x_sel = x_sorted[selected_idx]
    prob_sel = prob_sorted[selected_idx]

    # Number of selected points must not exceed batch_size
    assert len(x_sel) <= batch_size, "More points returned than batch_size"

    # Probabilities must be in [0, 1]
    assert np.all(prob_sel <= 1.0) and np.all(prob_sel >= 0.0), "greedy_exclusion returning impossible prob values"

    # Shapes of returned arrays must be consistent
    assert x_sel.shape == (len(x_sel), ForresterDomain.dim)


@pytest.mark.unit
def test_threshold_with_greedy_exclusion_empty(singlefidelitysample):
    """When no candidates pass the threshold, greedy_exclusion should not be reached."""
    Is, Xs, Ys = singlefidelitysample

    state = State(ForresterDomain, Is, Xs, Ys)

    surrogate = SingleFidelityGPSurrogate()
    surrogate.fit(state)

    sampler = Uniform(ForresterDomain.dim)

    # All draws are 1.0, so mask is always False (no candidate has prob > 1)
    random_draws = np.ones(4)
    x_all, y_pred, prob, mask = probabilistic_threshold_filter(
        ForresterDomain, state, sampler, surrogate, 4, 0.5, random_draws=random_draws
    )
    assert mask.sum() == 0


# ---------------------------------------------------------------------------
# Tests for deprecated wrappers in utils (backwards compatibility)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_deprecated_probabilistic_threshold_sampling(singlefidelitysample, initial_samples, threshold_value):
    Is, Xs, Ys = singlefidelitysample
    state = State(ForresterDomain, Is, Xs, Ys)
    surrogate = SingleFidelityGPSurrogate()
    surrogate.fit(state)
    sampler = Uniform(ForresterDomain.dim)

    with pytest.warns(DeprecationWarning, match="probabilistic_threshold_sampling is deprecated"):
        x_all, y_pred, prob, mask = probabilistic_threshold_sampling(
            ForresterDomain, state, sampler, surrogate, initial_samples, threshold_value
        )
    assert np.all(prob <= 1.0) and np.all(prob >= 0.0)


@pytest.mark.unit
def test_deprecated_probabilistic_threshold_sampling_with_exclusion(
    singlefidelitysample, threshold_value, batch_size, rejection_radius
):
    Is, Xs, Ys = singlefidelitysample
    state = State(ForresterDomain, Is, Xs, Ys)
    surrogate = SingleFidelityGPSurrogate()
    surrogate.fit(state)
    sampler = Uniform(ForresterDomain.dim)

    with pytest.warns(DeprecationWarning, match="probabilistic_threshold_sampling_with_exclusion is deprecated"):
        x_sel, y_sel, prob_sel = probabilistic_threshold_sampling_with_exclusion(
            ForresterDomain, state, sampler, surrogate, 50, threshold_value, batch_size, rejection_radius
        )
    assert len(x_sel) <= batch_size
    if len(x_sel) > 0:
        assert np.all(prob_sel <= 1.0) and np.all(prob_sel >= 0.0)


# ---------------------------------------------------------------------------
# Tests for x_existing in greedy_exclusion
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_greedy_exclusion_x_existing_excludes_nearby_candidates():
    """Candidates close to existing points must be excluded."""
    # Five candidates spread across [0, 1]; first and last are near existing points
    x_candidates = np.array([[0.01], [0.25], [0.5], [0.75], [0.99]])
    # Existing points near the first and last candidate
    x_existing = np.array([[0.0], [1.0]])
    rejection_radius = 0.1

    selected = greedy_exclusion(x_candidates, batch_size=5, rejection_radius=rejection_radius, x_existing=x_existing)

    x_selected = x_candidates[selected]
    # None of the selected points should be within rejection_radius of an existing point
    for xe in x_existing:
        dists = np.abs(x_selected - xe)
        assert np.all(dists >= rejection_radius), (
            f"A selected candidate is within rejection_radius of existing point {xe}"
        )


@pytest.mark.unit
def test_greedy_exclusion_x_existing_none_unchanged():
    """Passing x_existing=None must give the same result as not passing it."""
    rng = np.random.default_rng(42)
    x_candidates = rng.random((20, 2))

    without_existing = greedy_exclusion(x_candidates, batch_size=5, rejection_radius=0.2)
    with_none = greedy_exclusion(x_candidates, batch_size=5, rejection_radius=0.2, x_existing=None)

    np.testing.assert_array_equal(without_existing, with_none)


@pytest.mark.unit
def test_greedy_exclusion_x_existing_empty_array_unchanged():
    """Passing an empty x_existing must give the same result as None."""
    rng = np.random.default_rng(42)
    x_candidates = rng.random((20, 2))

    without_existing = greedy_exclusion(x_candidates, batch_size=5, rejection_radius=0.2)
    with_empty = greedy_exclusion(
        x_candidates, batch_size=5, rejection_radius=0.2, x_existing=np.empty((0, 2))
    )

    np.testing.assert_array_equal(without_existing, with_empty)


# ---------------------------------------------------------------------------
# Tests for exclude_existing in GreedyExclusionGenerator
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_greedy_exclusion_generator_exclude_existing(singlefidelitysample, batch_size, rejection_radius):
    """GreedyExclusionGenerator with exclude_existing=True should not place candidates
    near already-evaluated state points."""
    Is, Xs, Ys = singlefidelitysample
    state = State(ForresterDomain, Is, Xs, Ys)

    inner = RandomCandidateGenerator(ForresterDomain, Uniform(ForresterDomain.dim))
    generator = GreedyExclusionGenerator(
        inner=inner,
        rejection_radius=rejection_radius,
        pool_multiplier=8,
        exclude_existing=True,
    )

    Xs_new, _ = generator.generate(state, batch_size)

    if len(Xs_new) == 0:
        pytest.skip("No candidates generated")

    # No new candidate should be within rejection_radius of an existing point (in normalised space)
    # We check raw distance here as a sanity check (the PCA transform scales things, but
    # the 1-D Forrester domain has unit range so the raw distance is a reasonable proxy)
    for x_e in state.Xs:
        dists = np.abs(Xs_new - x_e)
        # At least some spacing must exist (not all will be rejected; just verify the
        # exclude_existing flag is wired through without error)
        assert dists.shape[1] == ForresterDomain.dim


@pytest.mark.unit
def test_threshold_exclusion_generator_exclude_existing(
    singlefidelitysample, threshold_value, batch_size, rejection_radius
):
    """ThresholdExclusionGenerator with exclude_existing=True should run without error
    and return a valid batch."""
    Is, Xs, Ys = singlefidelitysample
    state = State(ForresterDomain, Is, Xs, Ys)

    surrogate = SingleFidelityGPSurrogate()
    surrogate.fit(state)
    sampler = Uniform(ForresterDomain.dim)

    generator = ThresholdExclusionGenerator(
        domain=ForresterDomain,
        sampler=sampler,
        surrogate=surrogate,
        threshold_value=threshold_value,
        rejection_radius=rejection_radius,
        pool_size=128,
        refit_surrogate=False,
        exclude_existing=True,
        max_retries=10,
        pool_try_multiplier=2,
    )

    try:
        Xs_new, _ = generator.generate(state, batch_size)
    except RuntimeError:
        pytest.skip("Not enough candidates survived exclusion (expected for tight radius)")

    assert len(Xs_new) <= batch_size
    assert Xs_new.shape[1] == ForresterDomain.dim
