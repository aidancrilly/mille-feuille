from abc import ABC, abstractmethod
from typing import Sequence

import threading
from dataclasses import dataclass

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



# ---------------------------------------------------------------------------
# Resource accounting
# ---------------------------------------------------------------------------

@dataclass
class FidelityConfig:
    """Configuration for a single discrete fidelity level.

    Attributes:
        cores_required:   Number of cores needed to run a job at this fidelity.
        reserve:          If True the scheduler reserves cores so that jobs at
                          this fidelity are not starved by cheaper work.
        estimated_runtime: Optional runtime estimate in seconds (informational).
    """

    cores_required: int
    reserve: bool = False
    estimated_runtime: float | None = None


@dataclass
class Task:
    """A single simulation task.

    Attributes:
        index:          Unique run index.
        x:              Input parameter vector (1-D array).
        s:              Fidelity level (None for single-fidelity problems).
        cores_required: Number of cores this task will consume.
        reserve:        Whether this task belongs to a reserved fidelity.
    """

    index: int
    x: npt.NDArray
    s: int | None
    cores_required: int
    reserve: bool = False

class ResourceManager:
    """Thread-safe tracker for available compute cores."""

    def __init__(self, total_cores: int):
        self._total = total_cores
        self._used = 0
        self._lock = threading.Lock()

    @property
    def total(self) -> int:
        return self._total

    @property
    def used(self) -> int:
        with self._lock:
            return self._used

    @property
    def available(self) -> int:
        with self._lock:
            return self._total - self._used

    def allocate(self, cores: int) -> bool:
        """Try to allocate *cores*.  Returns True on success."""
        with self._lock:
            if self._used + cores <= self._total:
                self._used += cores
                return True
            return False

    def release(self, cores: int):
        """Return *cores* to the pool."""
        with self._lock:
            self._used = max(0, self._used - cores)

    def utilisation(self) -> float:
        """Current utilisation as a fraction in [0, 1]."""
        with self._lock:
            return self._used / self._total if self._total > 0 else 0.0