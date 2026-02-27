import numpy as np
import pytest
import pytest_cases
import torch
from botorch.acquisition import qLogExpectedImprovement, qUpperConfidenceBound
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from millefeuille.domain import FidelityDomain
from millefeuille.initialise import generate_initial_sample
from millefeuille.optimise import suggest_next_locations
from millefeuille.state import State
from millefeuille.surrogate import MultiFidelityGPSurrogate, SingleFidelityGPSurrogate, SingleFidelityRandomForestSurrogate
from millefeuille.utils import run_Bayesian_optimiser

from .conftest import (
    TEST_KERNEL,
    TEST_KERNEL_KWARGS,
    TEST_NUM_RESTARTS,
    TEST_RAW_SAMPLES,
    ForresterDomain,
    ForresterSampler,
    PythonForresterFunction,
)


@pytest_cases.fixture(params=[16])
def ntrain(request):
    return request.param


@pytest_cases.fixture(params=[1, 2])
def batch_size(request):
    return request.param


@pytest_cases.fixture(params=["qLogExpectedImprovement", "qUpperConfidenceBound"])
def generate_acq_function(request):
    if request.param == "qLogExpectedImprovement":
        return lambda surrogate, state: qLogExpectedImprovement(surrogate.model, state.best_value_transformed)
    elif request.param == "qUpperConfidenceBound":
        return lambda surrogate, state: qUpperConfidenceBound(surrogate.model, beta=1.0)


@pytest_cases.fixture(params=["qLogExpectedImprovement", "qUpperConfidenceBound"])
def generate_MF_acq_function(request):
    post_transform = ScalarizedPosteriorTransform(weights=torch.tensor([1.0]))
    if request.param == "qLogExpectedImprovement":
        return lambda surrogate, state: qLogExpectedImprovement(
            surrogate.model, state.best_value_transformed, posterior_transform=post_transform
        )
    elif request.param == "qUpperConfidenceBound":
        return lambda surrogate, state: qUpperConfidenceBound(
            surrogate.model, beta=1.0, posterior_transform=post_transform
        )


@pytest_cases.fixture()
def singlefidelitysample(ntrain):
    Is = np.arange(ntrain)
    Xs = np.linspace(0.0, 1.0, ntrain + 1)
    Xs = np.delete(Xs, np.argmin((Xs - 0.75725) ** 2)).reshape(ntrain, 1)
    f = PythonForresterFunction()
    _, Ys = f(Is, Xs)
    return Is, Xs, Ys, f


@pytest_cases.fixture()
def multifidelitysample(ntrain):
    Is = np.arange(ntrain)
    _rng = np.random.default_rng(seed=124)
    Ss = _rng.binomial(1, 0.5, size=ntrain).reshape(-1, 1)
    Xs, _ = generate_initial_sample(ForresterDomain, ForresterSampler(_rng), ntrain)
    f = PythonForresterFunction()
    _, Ys = f(Is, Xs, Ss)
    return Is, Xs, Ss, Ys, f


@pytest.mark.unit
def test_optimise_singlefidelity_GP(singlefidelitysample, batch_size, generate_acq_function):
    Is, Xs, Ys, f = singlefidelitysample

    state = State(ForresterDomain, Is, Xs, Ys)

    surrogate = SingleFidelityGPSurrogate(kernel=TEST_KERNEL, kernel_kwargs=TEST_KERNEL_KWARGS)
    surrogate.fit(state)

    best_y = float(state.best_value.item())

    acq_function = generate_acq_function(surrogate, state)

    X_next = suggest_next_locations(
        batch_size, state, acq_function, num_restarts=TEST_NUM_RESTARTS, raw_samples=TEST_RAW_SAMPLES
    )
    assert X_next.shape[0] == batch_size, "suggest_next_locations do not return batch_size candidates"
    assert X_next.shape[1] == Xs.shape[1], "suggest_next_locations candidates did not have same dimension as problem"

    I_next = np.amax(Is) + 1 + np.arange(batch_size)
    _, Y_next = f(I_next, X_next)

    state.update(I_next, X_next, Y_next)

    assert float(state.best_value.item()) >= best_y
    assert len(state.Ys) == len(Ys) + batch_size

    # Reset and use full wrapper
    initial_state = State(ForresterDomain, Is, Xs, Ys)
    surrogate = SingleFidelityGPSurrogate(kernel=TEST_KERNEL, kernel_kwargs=TEST_KERNEL_KWARGS)

    new_state = run_Bayesian_optimiser(
        1,
        batch_size,
        generate_acq_function,
        initial_state,
        surrogate,
        f,
        num_restarts=TEST_NUM_RESTARTS,
        raw_samples=TEST_RAW_SAMPLES,
    )

    assert np.isclose(new_state.best_value, state.best_value, rtol=1e-2)


@pytest.mark.unit
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_optimise_multifidelity_GP(multifidelitysample, batch_size, generate_MF_acq_function):
    Is, Xs, Ss, Ys, f = multifidelitysample

    ForresterFidelity = FidelityDomain(num_fidelities=len(np.unique(Ss)))

    state = State(ForresterDomain, Is, Xs, Ys, Ss=Ss, fidelity_domain=ForresterFidelity)

    surrogate = MultiFidelityGPSurrogate()
    surrogate.fit(state)

    best_y = float(state.best_value.item())

    acq_function = generate_MF_acq_function(surrogate, state)

    X_next, S_next = suggest_next_locations(
        batch_size, state, acq_function, num_restarts=TEST_NUM_RESTARTS, raw_samples=TEST_RAW_SAMPLES
    )
    assert X_next.shape[0] == batch_size, "suggest_next_locations do not return batch_size candidates"
    assert X_next.shape[1] == Xs.shape[1], "suggest_next_locations candidates did not have same dimension as problem"

    I_next = np.amax(Is) + 1 + np.arange(batch_size)
    _, Y_next = f(I_next, X_next, S_next)

    state.update(I_next, X_next, Y_next, S_next=S_next)

    assert float(state.best_value.item()) >= best_y
    assert len(state.Ys) == len(Ys) + batch_size

    # Reset and use full wrapper
    initial_state = State(ForresterDomain, Is, Xs, Ys, Ss=Ss, fidelity_domain=ForresterFidelity)
    surrogate = MultiFidelityGPSurrogate()

    new_state = run_Bayesian_optimiser(
        1,
        batch_size,
        generate_MF_acq_function,
        initial_state,
        surrogate,
        f,
        num_restarts=TEST_NUM_RESTARTS,
        raw_samples=TEST_RAW_SAMPLES,
    )

    assert np.isclose(new_state.best_value, state.best_value, rtol=1e-2)


@pytest.mark.unit
def test_optimise_singlefidelity_RF(singlefidelitysample, batch_size, generate_acq_function):
    Is, Xs, Ys, f = singlefidelitysample

    state = State(ForresterDomain, Is, Xs, Ys)

    surrogate = SingleFidelityRandomForestSurrogate(n_estimators=50)
    surrogate.fit(state)

    best_y = float(state.best_value.item())

    acq_function = generate_acq_function(surrogate, state)

    X_next = suggest_next_locations(
        batch_size, state, acq_function, num_restarts=TEST_NUM_RESTARTS, raw_samples=TEST_RAW_SAMPLES
    )
    assert X_next.shape[0] == batch_size, "suggest_next_locations do not return batch_size candidates"
    assert X_next.shape[1] == Xs.shape[1], "suggest_next_locations candidates did not have same dimension as problem"

    I_next = np.amax(Is) + 1 + np.arange(batch_size)
    _, Y_next = f(I_next, X_next)

    state.update(I_next, X_next, Y_next)

    assert float(state.best_value.item()) >= best_y
    assert len(state.Ys) == len(Ys) + batch_size

    # Reset and use full wrapper
    initial_state = State(ForresterDomain, Is, Xs, Ys)
    surrogate = SingleFidelityRandomForestSurrogate(n_estimators=50)

    new_state = run_Bayesian_optimiser(
        5,
        batch_size,
        generate_acq_function,
        initial_state,
        surrogate,
        f,
        num_restarts=TEST_NUM_RESTARTS,
        raw_samples=TEST_RAW_SAMPLES,
    )

    # since RF is non-deterministic and may not find the best point first time
    assert (
        np.isclose(new_state.best_value, state.best_value, atol=0.1)
        or new_state.best_value.item() > state.best_value.item()
    )
