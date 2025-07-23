import copy
import os
import warnings

import numpy as np
import pytest
import pytest_cases
from botorch.exceptions.warnings import OptimizationWarning
from millefeuille.initialise import generate_initial_sample
from millefeuille.state import State
from millefeuille.surrogate import SingleFidelityGPSurrogate

from .conftest import ForresterDomain, LowFidelityForresterMean, PythonForresterFunction, sampler


@pytest_cases.fixture(params=[5, 10, 20])
def ntrain(request):
    return request.param


@pytest_cases.fixture()
def singlefidelitysample(ntrain):
    Is = np.arange(ntrain)
    Xs, _ = generate_initial_sample(ForresterDomain, sampler, ntrain)
    f = PythonForresterFunction()
    Ys = f(Is, Xs)
    return Is, Xs, Ys


@pytest_cases.fixture(params=[2])
def ntest(request):
    return request.param


@pytest_cases.fixture()
def testXs(ntest):
    Xs, _ = generate_initial_sample(ForresterDomain, sampler, ntest)
    return Xs


@pytest.mark.unit
@pytest.mark.filterwarnings("ignore::botorch.exceptions.warnings.OptimizationWarning")
def test_singlefidelity_GP(singlefidelitysample, testXs):
    warnings.filterwarnings("ignore", category=OptimizationWarning)
    Is, Xs, Ys = singlefidelitysample

    state = State(ForresterDomain, Is, Xs, Ys)

    surrogate = SingleFidelityGPSurrogate()
    surrogate.init(state)
    surrogate.fit(state)
    testYs = surrogate.predict(state, testXs)

    surrogate.save("test.pth")

    second_surrogate = SingleFidelityGPSurrogate()
    second_surrogate.init(state)
    second_surrogate.load("test.pth", eval=True)
    os.remove("test.pth")
    second_testYs = second_surrogate.predict(state, testXs)

    assert np.isclose(testYs["mean"], second_testYs["mean"]).all(), (
        "Mean predictions diverged between saved and loaded surrogate model"
    )
    assert np.isclose(testYs["std"], second_testYs["std"]).all(), (
        "Std. dev. predictions diverged between saved and loaded surrogate model"
    )


@pytest.mark.unit
def test_singlefidelity_mean_module_GP(singlefidelitysample, testXs):
    warnings.filterwarnings("ignore", category=OptimizationWarning)
    Is, Xs, Ys = singlefidelitysample

    state = State(ForresterDomain, Is, Xs, Ys)

    output_scaler = copy.deepcopy(state.Y_scaler)
    output_scaler.training_override = True

    mean_surrogate = SingleFidelityGPSurrogate()
    mean_module = LowFidelityForresterMean(output_scaler)
    initial_mean_state_dict = mean_module.state_dict()
    mean_surrogate.init(state, mean_module=mean_module)
    mean_surrogate.fit(state)
    testYs = mean_surrogate.predict(state, testXs)

    mean_surrogate.save("test.pth")

    # Inspect mean_module trainable parameter
    for param_name, param in initial_mean_state_dict.items():
        if param_name in mean_surrogate.mean_module.state_dict():
            if "raw_constant" in param_name:
                assert not np.allclose(param, mean_surrogate.mean_module.state_dict()[param_name]), (
                    f"Parameter {param_name} has not been updated in the mean module after training"
                )
            else:
                assert np.allclose(param, mean_surrogate.mean_module.state_dict()[param_name]), (
                    f"Parameter {param_name} in mean module does not match initial state dict"
                )

    error_surrogate = SingleFidelityGPSurrogate()
    error_surrogate.init(state)
    # Check error is raised when don't use mean module
    with pytest.raises(Exception) as _:  # noqa
        error_surrogate.load("test.pth", eval=True)

    second_surrogate = SingleFidelityGPSurrogate()
    second_surrogate.init(state, mean_module=LowFidelityForresterMean(state.Y_scaler))
    second_surrogate.load("test.pth", eval=True)
    os.remove("test.pth")
    second_testYs = second_surrogate.predict(state, testXs)

    assert np.isclose(testYs["mean"], second_testYs["mean"]).all(), (
        "Mean predictions diverged between saved and loaded surrogate model"
    )
    assert np.isclose(testYs["std"], second_testYs["std"]).all(), (
        "Std. dev. predictions diverged between saved and loaded surrogate model"
    )
