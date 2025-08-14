import numpy as np
import pytest
import pytest_cases
from botorch.acquisition import qLogExpectedImprovement, qUpperConfidenceBound
from millefeuille.optimise import suggest_next_locations
from millefeuille.state import State
from millefeuille.surrogate import SingleFidelityGPSurrogate
from millefeuille.utils import run_Bayesian_optimiser

from .conftest import ForresterDomain, PythonForresterFunction


@pytest_cases.fixture(params=[5])
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
    Xs = np.append(np.linspace(0.0, 0.5, ntrain - 3), [0.7, 0.8, 1.0]).reshape(ntrain, 1)
    f = PythonForresterFunction()
    _, Ys = f(Is, Xs)
    return Is, Xs, Ys, f


@pytest.mark.unit
def test_optimise_singlefidelity_GP(singlefidelitysample, batch_size, generate_acq_function):
    Is, Xs, Ys, f = singlefidelitysample

    state = State(ForresterDomain, Is, Xs, Ys)

    surrogate = SingleFidelityGPSurrogate()
    surrogate.init_GP_model(state)

    best_y = float(state.best_value)
    print(best_y)

    acq_function = generate_acq_function(surrogate, state)

    X_next = suggest_next_locations(batch_size, state, surrogate, acq_function)
    assert X_next.shape[0] == batch_size, "suggest_next_locations do not return batch_size candidates"
    assert X_next.shape[1] == Xs.shape[1], "suggest_next_locations candidates did not have same dimension as problem"

    I_next = np.amax(Is) + 1 + np.arange(batch_size)
    _, Y_next = f(I_next, X_next)

    state.update(I_next, X_next, Y_next)

    assert float(state.best_value) > best_y
    assert len(state.Ys) == len(Ys) + batch_size

    # Reset and use full wrapper
    initial_state = State(ForresterDomain, Is, Xs, Ys)
    surrogate = SingleFidelityGPSurrogate()
    surrogate.init_GP_model(initial_state)

    new_state = run_Bayesian_optimiser(1, batch_size, generate_acq_function, initial_state, surrogate, f)

    assert np.isclose(new_state.best_value, state.best_value, rtol=1e-2)
