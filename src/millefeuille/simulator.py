from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
import numpy.typing as npt

"""
Defines a wrapper over your simulator which is ran on HPC resources
"""


class PythonSimulator(ABC):
    @abstractmethod
    def __call__(
        self, indices: npt.NDArray, Xs: npt.NDArray, Ss: None | npt.NDArray = None
    ) -> tuple[npt.NDArray | None, npt.NDArray]:
        pass


class Scheduler(ABC):
    """
    Abstract base class for scheduling and launching parallel simulations on HPC systems.
    """

    @abstractmethod
    def launch_jobs(self, exe: str, nprocs: Sequence[int], inputs: Sequence, indices: Sequence[str]):
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
    @abstractmethod
    def mpiexec(self) -> str:
        """Path to `mpiexec` or equivalent MPI launch command."""
        pass

    @property
    @abstractmethod
    def output_dir(self) -> str:
        """Directory to store stdout/stderr log files."""
        pass


class ExectuableSimulator(ABC):
    @property
    @abstractmethod
    def executable(self) -> str:
        pass

    @property
    @abstractmethod
    def nproc_per_fidelity(self) -> list[int]:
        pass

    def prepare_inputs(self, indices: npt.NDArray, Xs: npt.NDArray, Ss: None | npt.NDArray = None):
        if Ss is None:
            for i in range(len(indices)):
                self.single_prepare_inputs(indices[i], Xs[i, :])
        else:
            for i in range(len(indices)):
                self.single_prepare_inputs(indices[i], Xs[i, :], Ss[i, 0])

    @abstractmethod
    def single_prepare_inputs(self, index: int, x: npt.NDArray, s: int | None):
        pass

    @abstractmethod
    def launch(self, indices: npt.NDArray, Xs: npt.NDArray, scheduler: type[Scheduler], Ss: None | npt.NDArray = None):
        pass

    def postprocess(
        self, indices: npt.NDArray, Xs: npt.NDArray, Ss: None | npt.NDArray = None
    ) -> tuple[npt.NDArray | None, npt.NDArray]:
        Ps_list = []
        Ys_list = []

        for i in range(len(indices)):
            if Ss is None:
                Pi, Yi = self.single_postprocess(indices[i], Xs[i, :])
            else:
                Pi, Yi = self.single_postprocess(indices[i], Xs[i, :], Ss[i, 0])

            if Pi is not None:
                Ps_list.append(Pi)
            Ys_list.append(Yi)

        Ps = np.stack(Ps_list, axis=0) if Ps_list else None
        Ys = np.stack(Ys_list, axis=0)

        return Ps, Ys

    @abstractmethod
    def single_postprocess(self, index: int, x: npt.NDArray, s: int | None) -> tuple[npt.NDArray | None, npt.NDArray]:
        pass

    def __call__(
        self, indices: npt.NDArray, Xs: npt.NDArray, scheduler: type[Scheduler], Ss: None | npt.NDArray = None
    ):
        self.prepare_inputs(indices, Xs, Ss)
        self.launch(indices, Xs, scheduler, Ss)
        Ps, Ys = self.postprocess(indices, Xs, Ss)
        return Ps, Ys
