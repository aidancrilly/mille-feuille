"""
Defines a wrapper over your simulator which is ran on HPC resources
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy.typing as npt

class PythonSimulator(ABC):

    @abstractmethod
    def __call__(self, indices : npt.NDArray, Xs : npt.NDArray, Ss : None | npt.NDArray = None) -> tuple[Optional[npt.NDArray], npt.NDArray]:
        pass

class Scheduler(ABC):

    @property
    @abstractmethod
    def MPI_program(self) -> str:
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

    def prepare_inputs(self, indices : npt.NDArray, Xs : npt.NDArray, Ss : None | npt.NDArray = None):
        if(Ss is None):
            for i in range(len(indices)):
                self.single_prepare_inputs(indices[i],Xs[i,:])
        else:
            for i in range(len(indices)):
                self.single_prepare_inputs(indices[i],Xs[i,:],Ss[i,0])

    @abstractmethod
    def single_prepare_inputs(self, index : int, x : npt.NDArray, s : Optional[int]):
        pass

    @abstractmethod
    def launch(self, indices : npt.NDArray, Xs : npt.NDArray, scheduler : type[Scheduler], Ss : None | npt.NDArray = None):
        pass

    def postprocess(self, indices : npt.NDArray,  Xs : npt.NDArray, Ss : None | npt.NDArray = None) -> tuple[Optional[npt.NDArray], npt.NDArray]:
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
    def single_postprocess(self, index : int, x : npt.NDArray, s : Optional[int]) -> tuple[npt.NDArray | None, npt.NDArray]:
        pass

    def __call__(self, indices : npt.NDArray, Xs : npt.NDArray, scheduler : type[Scheduler], Ss : None | npt.NDArray = None):
        self.prepare_inputs(indices, Xs, Ss)
        self.launch(indices, Xs, scheduler, Ss)
        Ps, Ys = self.postprocess(indices, Xs, scheduler, Ss)
        return Ps, Ys
