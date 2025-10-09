import copy
import os
import warnings

import numpy as np
import pytest
import pytest_cases
import torch
import torch.nn as nn
from botorch.exceptions.warnings import OptimizationWarning
from millefeuille.initialise import generate_initial_sample
from millefeuille.state import State
from millefeuille.surrogate import BasePyTorchModel, SingleFidelityEnsembleSurrogate, SingleFidelityGPSurrogate

from .conftest import (
    TEST_KERNEL,
    TEST_KERNEL_KWARGS,
    ForresterDomain,
    ForresterSampler,
    LowFidelityForresterMean,
    PythonForresterFunction,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double


@pytest_cases.fixture(params=[5, 10, 20])
def ntrain(request):
    return request.param


@pytest_cases.fixture()
def singlefidelitysample(ntrain):
    Is = np.arange(ntrain)
    _rng = np.random.default_rng(seed=12345)
    Xs, _ = generate_initial_sample(ForresterDomain, ForresterSampler(_rng), ntrain)
    f = PythonForresterFunction()
    _, Ys = f(Is, Xs)
    return Is, Xs, Ys


@pytest_cases.fixture(params=[2])
def ntest(request):
    return request.param


@pytest_cases.fixture()
def testXs(ntest):
    _rng = np.random.default_rng(seed=12345)
    Xs, _ = generate_initial_sample(ForresterDomain, ForresterSampler(_rng), ntest)
    return Xs


@pytest.mark.unit
@pytest.mark.filterwarnings("ignore::botorch.exceptions.warnings.OptimizationWarning")
def test_singlefidelity_GP(singlefidelitysample, testXs):
    warnings.filterwarnings("ignore", category=OptimizationWarning)
    Is, Xs, Ys = singlefidelitysample

    state = State(ForresterDomain, Is, Xs, Ys)

    surrogate = SingleFidelityGPSurrogate(kernel=TEST_KERNEL, kernel_kwargs=TEST_KERNEL_KWARGS)
    surrogate.fit(state)
    testYs = surrogate.predict(state, testXs)

    surrogate.save("test.pth")

    second_surrogate = SingleFidelityGPSurrogate(kernel=TEST_KERNEL, kernel_kwargs=TEST_KERNEL_KWARGS)
    second_surrogate.init_GP_model(state)
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

    mean_module = LowFidelityForresterMean(output_scaler)
    initial_mean_state_dict = mean_module.state_dict()
    mean_surrogate = SingleFidelityGPSurrogate(
        mean_module=mean_module, kernel=TEST_KERNEL, kernel_kwargs=TEST_KERNEL_KWARGS
    )
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

    error_surrogate = SingleFidelityGPSurrogate(kernel=TEST_KERNEL, kernel_kwargs=TEST_KERNEL_KWARGS)
    error_surrogate.init_GP_model(state)
    # Check error is raised when don't use mean module
    with pytest.raises(Exception) as _:
        error_surrogate.load("test.pth", eval=True)

    second_surrogate = SingleFidelityGPSurrogate(
        mean_module=LowFidelityForresterMean(state.Y_scaler), kernel=TEST_KERNEL, kernel_kwargs=TEST_KERNEL_KWARGS
    )
    second_surrogate.init_GP_model(state)
    second_surrogate.load("test.pth", eval=True)
    os.remove("test.pth")
    second_testYs = second_surrogate.predict(state, testXs)

    assert np.isclose(testYs["mean"], second_testYs["mean"]).all(), (
        "Mean predictions diverged between saved and loaded surrogate model"
    )
    assert np.isclose(testYs["std"], second_testYs["std"]).all(), (
        "Std. dev. predictions diverged between saved and loaded surrogate model"
    )


class NNSurrogate(BasePyTorchModel):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 8, dtype=dtype, device=device),
            nn.Tanh(),
            nn.Linear(8, 8, dtype=dtype, device=device),
            nn.Tanh(),
            nn.Linear(8, 1, dtype=dtype, device=device),
        )

    @property
    def optimiser(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-2)

    @property
    def scheduler(self):
        class do_nothing(nn.Module):
            def __init__(self):
                super().__init__()

            def step(self):
                pass

        return do_nothing()

    @staticmethod
    def from_state_dict(state_dict):
        model = NNSurrogate()
        model.load_state_dict(state_dict)
        return model


@pytest.mark.unit
def test_singlefidelity_NNEnsemble(testXs):
    ntrain_NN = 50
    batch_size = 32
    nepochs = 400
    ensemble_size = 10

    Is = np.arange(ntrain_NN)
    _rng = np.random.default_rng(seed=12345)
    Xs, _ = generate_initial_sample(ForresterDomain, ForresterSampler(_rng), ntrain_NN)
    f = PythonForresterFunction()
    _, Ys = f(Is, Xs)

    state = State(ForresterDomain, Is, Xs, Ys)

    surrogate = SingleFidelityEnsembleSurrogate(
        ensemble_size=ensemble_size, model_base_class=NNSurrogate, training_epochs=nepochs, batch_size=batch_size
    )

    surrogate.fit(state)
    testYs = surrogate.predict(state, testXs)

    surrogate.save("test.pth")

    second_surrogate = SingleFidelityEnsembleSurrogate(
        ensemble_size=ensemble_size,
        model_base_class=NNSurrogate,
        training_epochs=nepochs,
        batch_size=batch_size,
        pretrained_file="test.pth",
    )
    os.remove("test.pth")

    second_testYs = second_surrogate.predict(state, testXs)

    assert np.isclose(testYs["mean"], second_testYs["mean"]).all(), (
        "Mean predictions diverged between saved and loaded surrogate model"
    )
    assert np.isclose(testYs["std"], second_testYs["std"]).all(), (
        "Std. dev. predictions diverged between saved and loaded surrogate model"
    )
