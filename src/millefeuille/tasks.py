"""
Defines data classes for holding task lists and available resources,
and a function for working through task lists asynchronously.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class TaskList:
    """
    Holds a list of pending simulation tasks.

    indices: run indices for each task (1D array, length N)
    Xs: input parameters for each task (shape: N x dim)
    Ss: fidelity levels for each task (shape: N x 1), or None for single-fidelity
    """

    indices: npt.NDArray
    Xs: npt.NDArray
    Ss: None | npt.NDArray = None

    def __post_init__(self):
        self.indices = np.asarray(self.indices).reshape(-1)
        self.Xs = np.atleast_2d(self.Xs)
        if self.Ss is not None:
            self.Ss = np.asarray(self.Ss).reshape(-1, 1)

    def __len__(self) -> int:
        return len(self.indices)

    def is_empty(self) -> bool:
        return len(self) == 0

    def pop(self, i: int = 0) -> tuple[int, npt.NDArray, int | None]:
        """
        Remove and return the task at position i as (index, x, s).
        s is None if Ss is None.
        """
        idx = int(self.indices[i])
        x = self.Xs[i, :]
        s = int(self.Ss[i, 0]) if self.Ss is not None else None

        self.indices = np.delete(self.indices, i)
        self.Xs = np.delete(self.Xs, i, axis=0)
        if self.Ss is not None:
            self.Ss = np.delete(self.Ss, i, axis=0)

        return idx, x, s

    def append(self, index: int | npt.NDArray, X: npt.NDArray, S: int | npt.NDArray | None = None):
        """
        Append one or more tasks to the list.

        index: int or 1D array of ints
        X: 1D (single task) or 2D (multiple tasks) array of inputs
        S: int, 1D/2D array, or None
        """
        new_indices = np.asarray(index).reshape(-1)
        if len(new_indices) == 0:
            return
        new_X = np.atleast_2d(X)

        self.indices = np.concatenate([self.indices, new_indices])
        self.Xs = np.concatenate([self.Xs, new_X], axis=0)

        if S is not None:
            new_S = np.asarray(S).reshape(-1, 1)
            self.Ss = np.concatenate([self.Ss, new_S], axis=0) if self.Ss is not None else new_S


@dataclass
class AvailableResources:
    """
    Holds information about available computational resources.

    ntotal_proc: total number of available processes/CPUs
    nproc_per_fidelity: list of processes required for each fidelity level,
                        e.g. [1, 2] means fidelity 0 needs 1 proc, fidelity 1 needs 2 procs
    """

    ntotal_proc: int
    nproc_per_fidelity: list[int]

    def __post_init__(self):
        if not self.nproc_per_fidelity:
            raise ValueError("nproc_per_fidelity must contain at least one entry")

    def nproc_for_task(self, s: int | None) -> int:
        """Return the number of processes required for fidelity s."""
        if s is None:
            return self.nproc_per_fidelity[0]
        return self.nproc_per_fidelity[int(s)]

    def can_fit_task(self, s: int | None, used_proc: int) -> bool:
        """Return True if a task at fidelity s fits within remaining resources."""
        return used_proc + self.nproc_for_task(s) <= self.ntotal_proc


def run_async_optimiser(
    task_list,
    resources,
    generate_next_tasks,
    state,
    surrogate,
    simulator,
    scheduler=None,
    retrain_interval: int = 1,
    csv_name: str | None = None,
):
    """
    Works through a task list asynchronously, keeping available resources
    occupied. As each task completes, its result is added to state, the
    surrogate is retrained at regular intervals, and the task list is renewed
    with new suggestions.

    Parameters:
        task_list: TaskList of initial pending tasks
        resources: AvailableResources describing available compute
        generate_next_tasks: callable(state, surrogate, n) -> TaskList
            Called every retrain_interval completions. Returns new tasks to
            add; return an empty TaskList (or None) to stop generating.
            n is the number of free resource slots (hint for how many to generate).
        state: current State (will be mutated in place)
        surrogate: surrogate model — retrained every retrain_interval completions
        simulator: PythonSimulator or ExectuableSimulator
        scheduler: Scheduler required when simulator is ExectuableSimulator
        retrain_interval: retrain surrogate and renew task list every N completions
        csv_name: optional filename to append results to CSV after each completion

    Returns:
        Updated State
    """
    from .simulator import ExectuableSimulator

    if isinstance(simulator, ExectuableSimulator) and scheduler is None:
        raise ValueError("A scheduler must be provided when using an ExectuableSimulator")

    completed_count = 0

    def _run_single(index, x, s):
        """Run one simulation task and return (P, Y)."""
        indices_1 = np.array([index])
        X_1 = x.reshape(1, -1)
        S_1 = np.array([[s]]) if s is not None else None
        if isinstance(simulator, ExectuableSimulator):
            return simulator(indices_1, X_1, scheduler, Ss=S_1)
        else:
            return simulator(indices_1, X_1, Ss=S_1)

    # Map from future -> (index, x, s, nproc_used)
    running: dict = {}

    def _used_procs() -> int:
        return sum(info[3] for info in running.values())

    def _fill_resources():
        """Launch tasks from task_list until no more tasks fit."""
        i = 0
        while i < len(task_list):
            s = int(task_list.Ss[i, 0]) if task_list.Ss is not None else None
            nproc = resources.nproc_for_task(s)
            if _used_procs() + nproc <= resources.ntotal_proc:
                idx, x, task_s = task_list.pop(i)
                future = executor.submit(_run_single, idx, x, task_s)
                running[future] = (idx, x, task_s, nproc)
                # Don't increment i — next task has shifted to position i
            else:
                i += 1

    max_workers = resources.ntotal_proc // min(resources.nproc_per_fidelity)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        _fill_resources()

        while running:
            # Block until any running task completes
            done_future = next(as_completed(list(running.keys())))
            idx, x, s, _nproc = running.pop(done_future)

            P, Y = done_future.result()

            # Add result to state
            S_next = np.array([[s]]) if s is not None else None
            state.update(
                np.array([idx]),
                X_next=x.reshape(1, -1),
                Y_next=Y,
                P_next=P,
                S_next=S_next,
            )
            completed_count += 1

            if csv_name is not None:
                state.to_csv(csv_name)

            # Retrain and renew task list at the configured interval
            if completed_count % retrain_interval == 0:
                surrogate.fit(state)
                free_procs = resources.ntotal_proc - _used_procs()
                n_free = free_procs // min(resources.nproc_per_fidelity)
                if n_free > 0:
                    new_tasks = generate_next_tasks(state, surrogate, n_free)
                    if new_tasks is not None and not new_tasks.is_empty():
                        if new_tasks.Ss is not None:
                            for j in range(len(new_tasks)):
                                task_list.append(
                                    new_tasks.indices[j], new_tasks.Xs[j], int(new_tasks.Ss[j, 0])
                                )
                        else:
                            for j in range(len(new_tasks)):
                                task_list.append(new_tasks.indices[j], new_tasks.Xs[j])

            # Fill any newly freed (or task-list-refilled) slots
            _fill_resources()

    return state
