import copy
import os
import warnings

import numpy as np
import pytest
import pytest_cases
import torch
import torch.nn as nn
from botorch.exceptions.warnings import OptimizationWarning
from millefeuille.domain import InputDomain
from millefeuille.initialise import generate_initial_sample
from millefeuille.state import State
from millefeuille.surrogate import (
    BasePyTorchModel,
    DeepKernelOptimiser,
    SingleFidelityDeepKernelGPSurrogate,
    SingleFidelityEnsembleSurrogate,
    SingleFidelityGPSurrogate,
)

from .conftest import ForresterDomain, ForresterSampler, LowFidelityForresterMean, PythonForresterFunction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double


@pytest_cases.fixture(params=[5, 10, 20])
def ntrain(request):
    return request.param


@pytest_cases.fixture()
def singlefidelitysample(ntrain):
    Is = np.arange(ntrain)
    Xs, _ = generate_initial_sample(ForresterDomain, ForresterSampler, ntrain)
    f = PythonForresterFunction()
    _, Ys = f(Is, Xs)
    return Is, Xs, Ys


@pytest_cases.fixture(params=[2])
def ntest(request):
    return request.param


@pytest_cases.fixture()
def testXs(ntest):
    Xs, _ = generate_initial_sample(ForresterDomain, ForresterSampler, ntest)
    return Xs


@pytest.mark.unit
@pytest.mark.filterwarnings("ignore::botorch.exceptions.warnings.OptimizationWarning")
def test_singlefidelity_GP(singlefidelitysample, testXs):
    warnings.filterwarnings("ignore", category=OptimizationWarning)
    Is, Xs, Ys = singlefidelitysample

    state = State(ForresterDomain, Is, Xs, Ys)

    surrogate = SingleFidelityGPSurrogate()
    surrogate.fit(state)
    testYs = surrogate.predict(state, testXs)

    surrogate.save("test.pth")

    second_surrogate = SingleFidelityGPSurrogate()
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

    mean_surrogate = SingleFidelityGPSurrogate()
    mean_module = LowFidelityForresterMean(output_scaler)
    initial_mean_state_dict = mean_module.state_dict()
    mean_surrogate.init_GP_model(state, mean_module=mean_module)
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
    error_surrogate.init_GP_model(state)
    # Check error is raised when don't use mean module
    with pytest.raises(Exception) as _:
        error_surrogate.load("test.pth", eval=True)

    second_surrogate = SingleFidelityGPSurrogate()
    second_surrogate.init_GP_model(state, mean_module=LowFidelityForresterMean(state.Y_scaler))
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
    Xs, _ = generate_initial_sample(ForresterDomain, ForresterSampler, ntrain_NN)
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


@pytest.mark.unit
def test_singlefidelity_deepkernel():
    Npts = 20
    Xs = np.linspace(-1.0, 1.0, Npts)
    Ys = np.heaviside(Xs, 0.0)

    input_domain = InputDomain(dim=1, b_low=np.array([-1.0]), b_up=np.array([1.0]), steps=np.array([0.0]))

    state = State(input_domain, np.arange(Npts), Xs, Ys)

    normal_surrogate = SingleFidelityGPSurrogate(noise_interval=(1e-8, 1.0))

    normal_surrogate.fit(state)

    deep_kernel_surrogate = SingleFidelityDeepKernelGPSurrogate(noise_interval=(1e-8, 1.0))

    deep_kernel = nn.Sequential(
        nn.Linear(1, 32, dtype=dtype, device=device),
        nn.Tanh(),
        nn.Linear(32, 32, dtype=dtype, device=device),
        nn.Tanh(),
        nn.Linear(32, 1, dtype=dtype, device=device),
    )

    from torch.optim import Adam

    optimiser = DeepKernelOptimiser(optimiser_method=Adam, num_epochs=200, learning_rate=0.025, verbose=True)

    deep_kernel_surrogate.fit(state, deep_kernel, optimiser)

    assert normal_surrogate.model.likelihood.noise.item() > deep_kernel_surrogate.model.likelihood.noise.item(), (
        "Deep kernel GP should learn lower noise on Heaviside data set..."
    )

    X_test = np.linspace(-1.0, 1.0, 100)
    y_pred_NS = normal_surrogate.predict(state, X_test.reshape(-1, 1))
    y_pred_DS = deep_kernel_surrogate.predict(state, X_test.reshape(-1, 1))
    y_truth = np.heaviside(X_test, 0.0)

    assert np.sum((y_truth - y_pred_NS["mean"].flatten()) ** 2) > np.sum(
        (y_truth - y_pred_DS["mean"].flatten()) ** 2
    ), "Deep kernel GP should learn lower MSE on Heaviside data set..."
