import numpy as np
import pytest
import pytest_cases
from millefeuille.initialise import generate_initial_sample

from .conftest import (
    CXXExecutableForrestorSimulator,
    ForresterDomain,
    FortranExecutableForrestorSimulator,
    PythonForresterFunction,
    ShellScheduler,
    sampler,
)

index_length = 4


@pytest_cases.fixture(params=[10])
def nsample(request):
    return request.param


@pytest_cases.fixture(params=[0.2, 0.5])
def multifidelitysample(nsample, request):
    Is = np.arange(nsample)
    Xs, _ = generate_initial_sample(ForresterDomain, sampler, nsample)
    f = PythonForresterFunction()
    Ss = np.random.binomial(1, request.param, size=nsample).reshape(-1, 1)
    _, Ys = f(Is, Xs, Ss)
    return Is, Xs, Ys, Ss


@pytest.mark.unit
def test_executable_simulators(multifidelitysample):
    Is, Xs, py_Ys, Ss = multifidelitysample

    # Scheduler
    scheduler = ShellScheduler()

    # Fortran
    fexe = FortranExecutableForrestorSimulator(index_length)
    fexe.prepare_inputs(Is, Xs, Ss)
    fexe.launch(Is, Xs, scheduler, Ss)
    _, f_Ys = fexe.postprocess(Is, Xs, Ss)

    fexe.cleanup(scheduler, Is)

    assert np.isclose(f_Ys.reshape(-1), py_Ys.reshape(-1)).all(), "Fortran executable Ys do not match equivalent Python"

    # C++
    cxxexe = CXXExecutableForrestorSimulator(index_length)
    cxxexe.prepare_inputs(Is, Xs, Ss)
    cxxexe.launch(Is, Xs, scheduler, Ss)
    _, cxx_Ys = cxxexe.postprocess(Is, Xs, Ss)

    cxxexe.cleanup(scheduler, Is)

    assert np.isclose(cxx_Ys.reshape(-1), py_Ys.reshape(-1)).all(), "C++ executable Ys do not match equivalent Python"
