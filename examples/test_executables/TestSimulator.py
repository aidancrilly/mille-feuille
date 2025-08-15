import json
import os

import f90nml
import numpy.typing as npt
from millefeuille.simulator import ExectuableSimulator, Scheduler


class ExecutableForrestorSimulator(ExectuableSimulator):
    """

    Base class which fortran and c++ implementations (below) will inherit from

    """

    def __init__(self, index_length):
        self.index_length = index_length

    @property
    def executable(self) -> str:
        pass

    @property
    def nproc_per_fidelity(self) -> list[int]:
        return [1, 2]

    def _get_input_file_name(self, index):
        pass

    def _get_output_file_name(self, index):
        pass

    def launch(self, indices: npt.NDArray, Xs: npt.NDArray, scheduler: type[Scheduler], Ss: None | npt.NDArray = None):
        file_paths = [(self._get_input_file_name(i), self._get_output_file_name(i)) for i in indices]
        str_indices = [str(i).zfill(self.index_length) for i in indices]
        if Ss is not None:
            nprocs = [self.nproc_per_fidelity[s] for s in Ss[:, 0]]
        else:
            nprocs = [self.nproc_per_fidelity[0] for _ in indices]
        scheduler.launch_jobs(self.executable, nprocs, file_paths, str_indices)

    def single_postprocess(self, index: int, x: npt.NDArray, s: int | None) -> tuple[npt.NDArray | None, npt.NDArray]:
        output_path = self._get_output_file_name(index)
        with open(output_path, "r") as f:
            Y = float(f.readline())
        return None, Y

    def cleanup(self, scheduler, indices):
        for index in indices:
            input_file = self._get_input_file_name(index)
            output_file = self._get_output_file_name(index)
            os.remove(input_file)
            os.remove(output_file)
        str_indices = [str(i).zfill(self.index_length) for i in indices]
        scheduler.cleanup(str_indices)


class CXXExecutableForrestorSimulator(ExecutableForrestorSimulator):
    @property
    def executable(self) -> str:
        return "./examples/test_executables/cxxmain"

    def _get_input_file_name(self, index):
        return f"cxxinput_{str(index).zfill(self.index_length)}.json"

    def _get_output_file_name(self, index):
        return f"cxxoutput_{str(index).zfill(self.index_length)}.txt"

    def single_prepare_inputs(self, index: int, x: npt.NDArray, s: int | None):
        input_path = self._get_input_file_name(index)
        json_dict = {"inputs": {"X": float(x[0]), "S": int(s)}}
        with open(input_path, "w") as fp:
            json.dump(json_dict, fp, indent=4)


class FortranExecutableForrestorSimulator(ExecutableForrestorSimulator):
    @property
    def executable(self) -> str:
        return "./examples/test_executables/fmain"

    def _get_input_file_name(self, index):
        return f"finput_{str(index).zfill(self.index_length)}.nml"

    def _get_output_file_name(self, index):
        return f"foutput_{str(index).zfill(self.index_length)}.txt"

    def single_prepare_inputs(self, index: int, x: npt.NDArray, s: int | None):
        input_path = self._get_input_file_name(index)
        nml_dict = {"inputs": {"X": x[0], "S": s}}
        nml = f90nml.Namelist(nml_dict)
        nml.write(input_path)
