import numpy as np
import pytest
import pytest_cases
import torch
from millefeuille.domain import FidelityDomain
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
def test_singlefidelity_probabilistic_threshold_sampling(singlefidelitysample, initial_samples, threshold_value):
    Is, Xs, Ys = singlefidelitysample

    state = State(ForresterDomain, Is, Xs, Ys)

    surrogate = SingleFidelityGPSurrogate()
    surrogate.fit(state)

    sampler = Uniform(ForresterDomain.dim)

    x_all, y_pred, prob, mask = probabilistic_threshold_sampling(
        ForresterDomain, state, sampler, surrogate, initial_samples, threshold_value
    )
    assert np.all(prob <= 1.0) and np.all(prob >= 0.0), (
        "probabilistic_threshold_sampling returning impossible prob values"
    )


@pytest.mark.unit
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_multifidelity_probabilistic_threshold_sampling(multifidelitysample, initial_samples, threshold_value):
    Is, Xs, Ss, Ys = multifidelitysample

    ForresterFidelity = FidelityDomain(num_fidelities=len(np.unique(Ss)))

    state = State(ForresterDomain, Is, Xs, Ys, Ss=Ss, fidelity_domain=ForresterFidelity)

    surrogate = MultiFidelityGPSurrogate(kernel=TEST_KERNEL, kernel_kwargs=TEST_KERNEL_KWARGS)
    surrogate.fit(state)

    sampler = Uniform(ForresterDomain.dim)

    with pytest.raises(ValueError):
        x_all, y_pred, prob, mask = probabilistic_threshold_sampling(
            ForresterDomain, state, sampler, surrogate, initial_samples, threshold_value, target_fidelity=None
        )

    for fidelity in range(2):
        x_all, y_pred, prob, mask = probabilistic_threshold_sampling(
            ForresterDomain, state, sampler, surrogate, initial_samples, threshold_value, target_fidelity=fidelity
        )
        assert np.all(prob <= 1.0) and np.all(prob >= 0.0), (
            "probabilistic_threshold_sampling returning impossible prob values"
        )


@pytest_cases.fixture(params=[3])
def batch_size(request):
    return request.param


@pytest_cases.fixture(params=[0.5])
def rejection_radius(request):
    return request.param


@pytest.mark.unit
def test_singlefidelity_probabilistic_threshold_sampling_with_exclusion(
    singlefidelitysample, threshold_value, batch_size, rejection_radius
):
    Is, Xs, Ys = singlefidelitysample

    state = State(ForresterDomain, Is, Xs, Ys)

    surrogate = SingleFidelityGPSurrogate()
    surrogate.fit(state)

    sampler = Uniform(ForresterDomain.dim)

    # Use a larger pool of candidates so the exclusion logic is exercised
    large_initial_samples = 50
    x_sel, y_sel, prob_sel = probabilistic_threshold_sampling_with_exclusion(
        ForresterDomain, state, sampler, surrogate, large_initial_samples, threshold_value, batch_size, rejection_radius
    )

    # Number of selected points must not exceed batch_size
    assert len(x_sel) <= batch_size, "More points returned than batch_size"

    # Probabilities must be in [0, 1]
    assert np.all(prob_sel <= 1.0) and np.all(prob_sel >= 0.0), (
        "probabilistic_threshold_sampling_with_exclusion returning impossible prob values"
    )

    # Shapes of returned arrays must be consistent
    assert x_sel.shape == (len(x_sel), ForresterDomain.dim)
    assert y_sel.shape == (len(x_sel),)
    assert prob_sel.shape == (len(x_sel),)


@pytest.mark.unit
def test_probabilistic_threshold_sampling_with_exclusion_empty(singlefidelitysample):
    """When no candidates pass the threshold, empty arrays should be returned."""
    Is, Xs, Ys = singlefidelitysample

    state = State(ForresterDomain, Is, Xs, Ys)

    surrogate = SingleFidelityGPSurrogate()
    surrogate.fit(state)

    sampler = Uniform(ForresterDomain.dim)

    # All draws are 1.0, so mask is always False (no candidate has prob > 1)
    random_draws = np.ones(4)
    x_sel, y_sel, prob_sel = probabilistic_threshold_sampling_with_exclusion(
        ForresterDomain, state, sampler, surrogate, 4, 0.5, 3, 0.5, random_draws=random_draws
    )
    assert len(x_sel) == 0
