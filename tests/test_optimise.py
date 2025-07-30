import numpy as np
import pytest
import pytest_cases
from botorch.acquisition import qLogExpectedImprovement, qUpperConfidenceBound
from millefeuille.initialise import generate_initial_sample
from millefeuille.optimise import suggest_next_locations
from millefeuille.state import State
from millefeuille.surrogate import SingleFidelityGPSurrogate

from .conftest import ForresterDomain, PythonForresterFunction, sampler


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
    Xs, _ = generate_initial_sample(ForresterDomain, sampler, ntrain)
    f = PythonForresterFunction()
    Ys = f(Is, Xs)
    return Is, Xs, Ys


@pytest.mark.unit
def test_optimise_singlefidelity_GP(singlefidelitysample, batch_size, generate_acq_function):
    Is, Xs, Ys = singlefidelitysample

    state = State(ForresterDomain, Is, Xs, Ys)

    surrogate = SingleFidelityGPSurrogate()
    surrogate.init_GP_model(state)

    acq_function = generate_acq_function(surrogate, state)

    X_next = suggest_next_locations(batch_size, state, surrogate, acq_function)

    assert X_next.shape[0] == batch_size, "suggest_next_locations do not return batch_size candidates"
    assert X_next.shape[1] == Xs.shape[1], "suggest_next_locations candidates did not have same dimension as problem"
