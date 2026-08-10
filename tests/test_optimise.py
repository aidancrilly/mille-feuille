import numpy as np
import pytest
import pytest_cases
import torch
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from millefeuille.acquisition import get_qLogEHVI_acq, get_qLogEI_acq, get_qLogEI_MF_acq, get_qUCB_acq, get_qUCB_MF_acq
from millefeuille.domain import FidelityDomain, InputDomain
from millefeuille.initialise import generate_initial_sample
from millefeuille.optimise import suggest_next_locations
from millefeuille.state import State
from millefeuille.surrogate import (
    MultiFidelityGPSurrogate,
    MultiObjectiveGPSurrogate,
    SingleFidelityGPSurrogate,
    SingleFidelityRandomForestSurrogate,
)
from millefeuille.utils import run_Bayesian_optimiser

from .conftest import (
    TEST_KERNEL,
    TEST_KERNEL_KWARGS,
    TEST_NUM_RESTARTS,
    TEST_RAW_SAMPLES,
    ForresterDomain,
    ForresterSampler,
    PythonBiObjectiveFunction,
    PythonForresterFunction,
)


@pytest_cases.fixture(params=[16])
def ntrain(request):
    return request.param


@pytest_cases.fixture(params=[1, 2])
def batch_size(request):
    return request.param


@pytest_cases.fixture(params=["qLogEI", "qUCB"])
def generate_acq_function(request):
    if request.param == "qLogEI":
        return get_qLogEI_acq
    elif request.param == "qUCB":
        return get_qUCB_acq


@pytest_cases.fixture(params=["qLogEI", "qUCB"])
def generate_MF_acq_function(request):
    if request.param == "qLogEI":
        return get_qLogEI_MF_acq
    elif request.param == "qUCB":
        return get_qUCB_MF_acq


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
def test_optimise_singlefidelity_GP_fixed_features(singlefidelitysample, batch_size, generate_acq_function):
    Is, Xs, Ys, f = singlefidelitysample

    # Use a 2D domain with non-unit bounds to test real-unit fixed_features transformation
    domain_2d = InputDomain(dim=2, b_low=np.array([0.0, 0.0]), b_up=np.array([1.0, 10.0]), steps=np.array([0.0, 0.0]))
    Xs_2d = np.hstack([Xs, np.zeros_like(Xs)])
    state_2d = State(domain_2d, Is, Xs_2d, Ys)

    surrogate = SingleFidelityGPSurrogate(kernel=TEST_KERNEL, kernel_kwargs=TEST_KERNEL_KWARGS)
    surrogate.fit(state_2d)

    acq_function = generate_acq_function(surrogate, state_2d)

    # Fix the second feature (index 1) to 5.0 in real units (= 0.5 in normalised units)
    fixed_features = {1: 5.0}
    X_next = suggest_next_locations(
        batch_size,
        state_2d,
        acq_function,
        num_restarts=TEST_NUM_RESTARTS,
        raw_samples=TEST_RAW_SAMPLES,
        fixed_features=fixed_features,
    )
    assert X_next.shape[0] == batch_size, "suggest_next_locations do not return batch_size candidates"
    assert X_next.shape[1] == Xs_2d.shape[1], "suggest_next_locations candidates did not have same dimension as problem"
    assert np.allclose(X_next[:, 1], 5.0), "fixed feature should be fixed at the specified real-unit value"


@pytest.mark.unit
def test_optimise_fixed_features_raises_for_multifidelity(multifidelitysample, generate_MF_acq_function):
    Is, Xs, Ss, Ys, f = multifidelitysample

    ForresterFidelity = FidelityDomain(num_fidelities=len(np.unique(Ss)))
    state = State(ForresterDomain, Is, Xs, Ys, Ss=Ss, fidelity_domain=ForresterFidelity)

    surrogate = MultiFidelityGPSurrogate()
    surrogate.fit(state)

    acq_function = generate_MF_acq_function(surrogate, state)

    with pytest.raises(ValueError, match="fixed_features is only supported for single fidelity optimisation"):
        suggest_next_locations(
            1,
            state,
            acq_function,
            num_restarts=TEST_NUM_RESTARTS,
            raw_samples=TEST_RAW_SAMPLES,
            fixed_features={0: 0.5},
        )


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


@pytest_cases.fixture()
def multiobjective_sample(ntrain):
    Is = np.arange(ntrain)
    Xs = np.linspace(0.0, 1.0, ntrain).reshape(-1, 1)
    f = PythonBiObjectiveFunction()
    _, Ys = f(Is, Xs)
    return Is, Xs, Ys, f


@pytest.mark.unit
def test_optimise_multiobjective_GP(multiobjective_sample, batch_size):
    Is, Xs, Ys, f = multiobjective_sample

    state = State(ForresterDomain, Is, Xs, Ys)

    surrogate = MultiObjectiveGPSurrogate(kernel=TEST_KERNEL, kernel_kwargs=TEST_KERNEL_KWARGS)
    surrogate.fit(state)

    acq_function = get_qLogEHVI_acq(surrogate, state)

    # 1. Return type
    assert isinstance(acq_function, qLogExpectedHypervolumeImprovement), (
        "get_qLogEHVI_acq did not return qLogExpectedHypervolumeImprovement"
    )

    # 2. Default ref_point matches state.worst_value_transformed
    expected_ref = torch.as_tensor(state.worst_value_transformed).flatten().double()
    assert torch.allclose(acq_function.ref_point.double(), expected_ref), (
        "ref_point does not match state.worst_value_transformed"
    )

    # 3. Candidate shape
    X_next = suggest_next_locations(
        batch_size, state, acq_function, num_restarts=TEST_NUM_RESTARTS, raw_samples=TEST_RAW_SAMPLES
    )
    assert X_next.shape == (batch_size, Xs.shape[1]), "suggest_next_locations returned unexpected shape"

    # 4. Hypervolume is non-decreasing after one BO step
    # Use a fixed reference point safely below all objective values in real space:
    # f1, f2 in [-(0.8)^2, 0] = [-0.64, 0], so [-1.0, -1.0] is a valid lower bound
    hv_ref = torch.tensor([-1.0, -1.0], dtype=torch.double)

    Y_before = torch.tensor(state.Ys, dtype=torch.double)
    hv_before = Hypervolume(ref_point=hv_ref).compute(Y_before[is_non_dominated(Y_before)])

    I_next = np.amax(Is) + 1 + np.arange(batch_size)
    _, Y_next = f(I_next, X_next)
    state.update(I_next, X_next, Y_next)

    Y_after = torch.tensor(state.Ys, dtype=torch.double)
    hv_after = Hypervolume(ref_point=hv_ref).compute(Y_after[is_non_dominated(Y_after)])

    assert hv_after >= hv_before, f"Hypervolume decreased after BO step: {hv_before:.6f} -> {hv_after:.6f}"
