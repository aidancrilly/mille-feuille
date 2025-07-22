import json
import os
import subprocess
from typing import Sequence

import f90nml
import numpy as np
import numpy.typing as npt
import torch
from gpytorch.means import Mean
from millefeuille.domain import InputDomain
from millefeuille.simulator import ExectuableSimulator, PythonSimulator, Scheduler

sampler = np.random.default_rng(seed=123456)


ForresterDomain = InputDomain(dim=1, b_low=np.array([0.0]), b_up=np.array([1.0]), steps=np.array([0.0]))


class PythonForresterFunction(PythonSimulator):
    """
    Multi-fidelity Forrestor function (negated for maximisation)

    """

    def f(self, Xs):
        ys = (6 * Xs - 2) ** 2 * np.sin(12 * Xs + 4)
        return ys

    def ABC_values(self, Ss):
        if Ss is not None:
            A = 1.0 - (1 - Ss) * 0.5
            B = (1 - Ss) * 10.0
            C = (1 - Ss) * 5.0
        else:
            A = 1.0
            B = 0.0
            C = 0.0
        return A, B, C

    def __call__(self, indices, Xs, Ss=None):
        A, B, C = self.ABC_values(Ss)
        return -(A * self.f(Xs) + B * (Xs - 0.5) + C)


class LowFidelityForresterMean(Mean):
    """

    A mean function that returns the low fidelity estimator of the Forrestor

    """

    def f(self, Xs):
        ys = (6 * Xs - 2) ** 2 * torch.sin(12 * Xs + 4)
        return ys

    def ABC_values(self, Ss):
        if Ss is not None:
            A = 1.0 - (1 - Ss) * 0.5
            B = (1 - Ss) * 10.0
            C = (1 - Ss) * 5.0
        else:
            A = 1.0
            B = 0.0
            C = 0.0
        return A, B, C

    def get_low_fid_ABC(self):
        return self.ABC_values(0.0)

    def forward(self, Xs):
        A, B, C = self.get_low_fid_ABC()
        return -(A * self.f(Xs) + B * (Xs - 0.5) + C).squeeze(-1)


class ShellScheduler(Scheduler):
    def _is_sequence(self, seq):
        return type(seq) in [list, tuple]

    @property
    def mpiexec(self) -> str:
        """Path to `mpiexec` or equivalent MPI launch command."""
        return "mpiexec"

    @property
    def output_dir(self) -> str:
        """Directory to store stdout/stderr log files."""
        return "."

    def launch_jobs(self, exe: str, nprocs: Sequence[int], inputs: Sequence, indices: Sequence[str]):
        """
        Launches simulation jobs in parallel.

        Parameters:
            exe: Path to the MPI-enabled executable
            nprocs: Number of processes per job
            inputs: List of input file paths
            indices: List of run indices (for logging/output naming)
        """
        Nbatch = len(inputs)

        processes = []
        outs = []
        errs = []
        for ibatch, (index, nproc, job_inputs) in enumerate(zip(indices, nprocs, inputs, strict=False)):
            output_file = os.path.join(self.output_dir, f"std_output_{index}.txt")
            error_file = os.path.join(self.output_dir, f"std_error_{index}.txt")

            out = open(output_file, "w")
            err = open(error_file, "w")

            exe_cmd = f"{self.mpiexec} -n {nproc} -wdir {self.output_dir} {exe}"
            cmd_line_args = " "
            if job_inputs is not None:
                if self._is_sequence(job_inputs):
                    for job_input in job_inputs:
                        cmd_line_args += f'"{job_input}" '
                else:
                    cmd_line_args = f'"{job_inputs}"'
            exe_cmd += cmd_line_args

            print(f"Launching job {ibatch + 1}/{Nbatch}: {exe_cmd}")
            proc = subprocess.Popen(exe_cmd, stdout=out, stderr=err, shell=True)
            processes.append(proc)
            outs.append(out)
            errs.append(err)

        for proc, out, err in zip(processes, outs, errs, strict=False):
            proc.wait()
            out.close()
            err.close()

    def cleanup(self, indices):
        for index in indices:
            output_file = os.path.join(self.output_dir, f"std_output_{index}.txt")
            error_file = os.path.join(self.output_dir, f"std_error_{index}.txt")
            os.remove(output_file)
            os.remove(error_file)


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
        return "./tests/test_exe/cxxmain"

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
        return "./tests/test_exe/fmain"

    def _get_input_file_name(self, index):
        return f"finput_{str(index).zfill(self.index_length)}.nml"

    def _get_output_file_name(self, index):
        return f"foutput_{str(index).zfill(self.index_length)}.txt"

    def single_prepare_inputs(self, index: int, x: npt.NDArray, s: int | None):
        input_path = self._get_input_file_name(index)
        nml_dict = {"inputs": {"X": x[0], "S": s}}
        nml = f90nml.Namelist(nml_dict)
        nml.write(input_path)
