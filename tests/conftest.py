from typing import Sequence

import numpy as np
import numpy.typing as npt
from millefeuille.domain import InputDomain
from millefeuille.simulator import ExectuableSimulator, PythonSimulator, Scheduler

sampler = np.random.default_rng(seed=12345)


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


class ExecutableForrestorSimulator(ExectuableSimulator):
    @property
    def executable(self) -> str:
        return "a.out"

    @property
    def nproc_per_fidelity(self) -> list[int]:
        return [1, 1]

    def single_prepare_inputs(self, index: int, x: npt.NDArray, s: int | None):
        pass

    def launch(self, indices: npt.NDArray, Xs: npt.NDArray, scheduler: type[Scheduler], Ss: None | npt.NDArray = None):
        pass

    def single_postprocess(self, index: int, x: npt.NDArray, s: int | None) -> tuple[npt.NDArray | None, npt.NDArray]:
        pass


class ShellScheduler(Scheduler):
    def launch_jobs(self, exe: str, nproc: int, inputs: Sequence[str], indices: Sequence[str]):
        """
        Launches simulation jobs in parallel.

        Parameters:
            exe: Path to the MPI-enabled executable
            nproc: Number of processes per job
            inputs: List of input file paths
            indices: List of run indices (for logging/output naming)
        """
        pass

    @property
    def mpiexec(self) -> str:
        """Path to `mpiexec` or equivalent MPI launch command."""
        return "mpiexec"

    @property
    def output_dir(self) -> str:
        """Directory to store stdout/stderr log files."""
        return "."
