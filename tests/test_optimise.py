import numpy as np
import pytest
import pytest_cases
from botorch.acquisition import qLogExpectedImprovement, qUpperConfidenceBound
from millefeuille.optimise import suggest_next_locations
from millefeuille.state import State
from millefeuille.surrogate import SingleFidelityGPSurrogate
from millefeuille.utils import run_Bayesian_optimiser

from .conftest import (
    TEST_KERNEL,
    TEST_KERNEL_KWARGS,
    TEST_NUM_RESTARTS,
    TEST_RAW_SAMPLES,
    ForresterDomain,
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


@pytest_cases.fixture()
def singlefidelitysample(ntrain):
    Is = np.arange(ntrain)
    Xs = np.linspace(0.0, 1.0, ntrain + 1)
    Xs = np.delete(Xs, np.argmin((Xs - 0.75725) ** 2)).reshape(ntrain, 1)
    f = PythonForresterFunction()
    _, Ys = f(Is, Xs)
    return Is, Xs, Ys, f


@pytest.mark.unit
def test_optimise_singlefidelity_GP(singlefidelitysample, batch_size, generate_acq_function):
    Is, Xs, Ys, f = singlefidelitysample

    state = State(ForresterDomain, Is, Xs, Ys)

    surrogate = SingleFidelityGPSurrogate(kernel=TEST_KERNEL, kernel_kwargs=TEST_KERNEL_KWARGS)
    surrogate.fit(state)

    best_y = float(state.best_value)

    acq_function = generate_acq_function(surrogate, state)

    X_next = suggest_next_locations(
        batch_size, state, acq_function, num_restarts=TEST_NUM_RESTARTS, raw_samples=TEST_RAW_SAMPLES
    )
    assert X_next.shape[0] == batch_size, "suggest_next_locations do not return batch_size candidates"
    assert X_next.shape[1] == Xs.shape[1], "suggest_next_locations candidates did not have same dimension as problem"

    I_next = np.amax(Is) + 1 + np.arange(batch_size)
    _, Y_next = f(I_next, X_next)

    state.update(I_next, X_next, Y_next)

    assert float(state.best_value) >= best_y
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
