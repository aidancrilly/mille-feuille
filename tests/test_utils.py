import numpy as np
import pytest
import pytest_cases
import torch
from millefeuille.domain import FidelityDomain
from millefeuille.initialise import generate_initial_sample
from millefeuille.state import State
from millefeuille.surrogate import MultiFidelityGPSurrogate, SingleFidelityGPSurrogate
from millefeuille.utils import probabilistic_threshold_sampling

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
    Xs, _ = generate_initial_sample(ForresterDomain, ForresterSampler, ntrain)
    f = PythonForresterFunction()
    _, Ys = f(Is, Xs)
    return Is, Xs, Ys


@pytest_cases.fixture()
def multifidelitysample(ntrain):
    Is = np.arange(ntrain)
    Ss = np.random.binomial(1, 0.5, size=ntrain).reshape(-1, 1)
    Xs, _ = generate_initial_sample(ForresterDomain, ForresterSampler, ntrain)
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
