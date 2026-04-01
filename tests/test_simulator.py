import os
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import pytest_cases
from millefeuille.initialise import generate_initial_sample

from .conftest import (
    ForresterDomain,
    ForresterSampler,
    PythonForresterFunction,
)

sys.path.append(os.path.join(os.path.dirname(__file__), "../examples/test_executables/"))

from TestScheduler import ShellScheduler
from TestSimulator import CXXExecutableForrestorSimulator, FortranExecutableForrestorSimulator

index_length = 4


def _replace_inputs(input_file, parameter_dict):
    """
    Standalone version of the replace_inputs pattern used in examples/loops/Simulator.py.
    Uses direct file I/O instead of fileinput to be thread-safe.
    """
    with open(input_file, "r") as f:
        lines = f.readlines()
    dir_name = os.path.dirname(input_file) or "."
    with tempfile.NamedTemporaryFile(mode="w", dir=dir_name, delete=False, suffix=".tmp") as tmp:
        tmp_path = tmp.name
        for line in lines:
            for key, value in parameter_dict.items():
                if key.lower() == line.split("=")[0].strip().lower():
                    line = f"\t{key} = {value} \n"
            tmp.write(line)
    os.replace(tmp_path, input_file)


@pytest_cases.fixture(params=[10])
def nsample(request):
    return request.param


@pytest_cases.fixture(params=[0.2, 0.5])
def multifidelitysample(nsample, request):
    Is = np.arange(nsample)
    _rng = np.random.default_rng(seed=12345)
    Xs, _ = generate_initial_sample(ForresterDomain, ForresterSampler(_rng), nsample)
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


@pytest.fixture
def namelist_template(tmp_path):
    """Create a simple namelist-style template file for replace_inputs tests."""
    template = tmp_path / "template.nml"
    template.write_text("\tindex = 0000 \n\toutput_dir = './default/' \n\tresolution = 128 \n")
    return template


@pytest.mark.unit
def test_replace_inputs_sequential(tmp_path, namelist_template):
    """Test that replace_inputs works correctly when called multiple times in sequence."""
    for i in range(5):
        target = tmp_path / f"input_{i}.nml"
        shutil.copy(namelist_template, target)
        _replace_inputs(
            str(target),
            {"index": str(i).zfill(4), "output_dir": f"'./run_{i}/'"},
        )

        content = target.read_text()
        assert f"index = {str(i).zfill(4)}" in content
        assert f"output_dir = './run_{i}/'" in content
        assert "resolution = 128" in content


@pytest.mark.unit
def test_replace_inputs_concurrent(tmp_path, namelist_template):
    """Test that replace_inputs is safe when called from multiple threads on different files."""
    n_files = 10
    targets = []
    for i in range(n_files):
        target = tmp_path / f"input_{i}.nml"
        shutil.copy(namelist_template, target)
        targets.append(target)

    def _replace(i):
        _replace_inputs(
            str(targets[i]),
            {"index": str(i).zfill(4), "output_dir": f"'./run_{i}/'"},
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(_replace, range(n_files)))

    for i in range(n_files):
        content = targets[i].read_text()
        assert f"index = {str(i).zfill(4)}" in content
        assert f"output_dir = './run_{i}/'" in content
        assert "resolution = 128" in content
