"""
Asynchronous scheduler for heterogeneous-fidelity simulations.

Replaces the synchronous batch -> wait -> analyse pattern with continuous
scheduling that maximises core utilisation using ThreadPoolExecutor.

Key components:
    FidelityConfig  - per-fidelity resource/reservation settings
    Task            - lightweight (index, X, S) task descriptor
    ResourceManager - thread-safe core accounting
    AsyncScheduler  - main scheduling loop with reservation + backfill
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .simulator import ExectuableSimulator, PythonSimulator, Scheduler

__all__ = ["FidelityConfig", "Task", "ResourceManager", "AsyncScheduler"]

logger = logging.getLogger("millefeuille.scheduler")


# ---------------------------------------------------------------------------
# Lightweight data classes
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


# ---------------------------------------------------------------------------
# Resource accounting
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Asynchronous scheduler
# ---------------------------------------------------------------------------


class AsyncScheduler:
    """Asynchronous, resource-aware scheduler for simulation tasks.

    Uses a ``ThreadPoolExecutor`` to launch and wait for jobs concurrently.
    The main scheduling loop runs in the calling thread and is responsible for:

    * checking completed futures,
    * updating ``State`` with results,
    * optionally generating new candidates via a user callback,
    * launching new jobs when resources allow.

    A **reservation + backfill** strategy ensures expensive (high-core)
    fidelities are not starved by cheaper work.

    Parameters:
        simulator:        Simulator instance (``ExectuableSimulator`` or
                          ``PythonSimulator``).
        resource_manager: ``ResourceManager`` tracking available cores.
        scheduler:        ``Scheduler`` for launching subprocess jobs
                          (required when *simulator* is an
                          ``ExectuableSimulator``).
        fidelity_configs: Optional mapping ``{fidelity_level: FidelityConfig}``.
                          Falls back to ``simulator.nproc_per_fidelity`` for
                          ``ExectuableSimulator`` or 1 core per task otherwise.
        max_workers:      Maximum threads in the executor (default 16).
        poll_interval:    Seconds between scheduling loop iterations (default 0.5).
    """

    def __init__(
        self,
        simulator: ExectuableSimulator | PythonSimulator,
        resource_manager: ResourceManager,
        scheduler: Scheduler | None = None,
        fidelity_configs: dict[int, FidelityConfig] | None = None,
        max_workers: int = 16,
        poll_interval: float = 0.5,
    ):
        if isinstance(simulator, ExectuableSimulator) and scheduler is None:
            raise ValueError("A Scheduler instance is required when using an ExectuableSimulator")

        self._scheduler = scheduler
        self._simulator = simulator
        self._resources = resource_manager
        self._fidelity_configs = fidelity_configs or {}
        self._max_workers = max_workers
        self._poll_interval = poll_interval
        self._pending_tasks: list[Task] = []

    # -- helpers ------------------------------------------------------------

    def _get_fidelity_config(self, s: int | None) -> FidelityConfig:
        """Return the ``FidelityConfig`` for fidelity *s*."""
        if s is not None and s in self._fidelity_configs:
            return self._fidelity_configs[s]
        # Fall back to simulator metadata
        if isinstance(self._simulator, ExectuableSimulator):
            s_idx = s if s is not None else 0
            cores = self._simulator.nproc_per_fidelity[s_idx]
            return FidelityConfig(cores_required=cores)
        return FidelityConfig(cores_required=1)

    def create_tasks(
        self,
        indices: npt.NDArray,
        Xs: npt.NDArray,
        Ss: npt.NDArray | None = None,
    ) -> list[Task]:
        """Build ``Task`` objects from arrays of indices, inputs, and (optional) fidelities."""
        tasks = []
        for i in range(len(indices)):
            s = int(Ss[i, 0]) if Ss is not None else None
            cfg = self._get_fidelity_config(s)
            tasks.append(
                Task(
                    index=int(indices[i]),
                    x=Xs[i, :].copy(),
                    s=s,
                    cores_required=cfg.cores_required,
                    reserve=cfg.reserve,
                )
            )
        return tasks

    def add_tasks(self, tasks: list[Task]):
        """Append *tasks* to the pending queue."""
        self._pending_tasks.extend(tasks)

    # -- scheduling strategy ------------------------------------------------

    def _schedule_next(self) -> list[Task]:
        """Select pending tasks to launch using reservation + backfill.

        Strategy
        --------
        1. Sort pending tasks by ``cores_required`` descending.
        2. If any *reserved* task is pending, set aside enough cores to
           accommodate the most expensive one.
        3. Launch reserved tasks first when resources suffice.
        4. Backfill remaining capacity with non-reserved tasks, provided
           the reservation is still honoured.
        """
        if not self._pending_tasks:
            return []

        available = self._resources.available
        sorted_pending = sorted(self._pending_tasks, key=lambda t: -t.cores_required)

        # Determine reservation (cores set aside for most-expensive reserved task)
        reserved_cores = 0
        for t in sorted_pending:
            if t.reserve:
                reserved_cores = t.cores_required
                break

        to_launch: list[Task] = []
        remaining = available

        for task in sorted_pending:
            if task.cores_required > remaining:
                continue

            if task.reserve:
                to_launch.append(task)
                remaining -= task.cores_required
                # Update reservation for the *next* most-expensive reserved task
                still_reserved = [t for t in sorted_pending if t.reserve and t not in to_launch]
                reserved_cores = still_reserved[0].cores_required if still_reserved else 0
            elif remaining - task.cores_required >= reserved_cores:
                # Backfill: only if reservation is still honoured
                to_launch.append(task)
                remaining -= task.cores_required

        for task in to_launch:
            self._pending_tasks.remove(task)

        return to_launch

    # -- task execution (runs in worker thread) -----------------------------

    def _run_single_task(self, task: Task):
        """Prepare, launch, wait, and postprocess a single task.

        This method blocks until the simulation subprocess completes and is
        intended to run inside a ``ThreadPoolExecutor`` worker thread.
        """
        index_arr = np.array([task.index])
        x_arr = task.x.reshape(1, -1)
        s_arr = np.array([[task.s]]) if task.s is not None else None

        if isinstance(self._simulator, ExectuableSimulator):
            self._simulator.single_prepare_inputs(task.index, task.x, task.s)
            self._simulator.launch(index_arr, x_arr, self._scheduler, s_arr)
            P, Y = self._simulator.single_postprocess(task.index, task.x, task.s)
        else:
            # PythonSimulator — call directly
            P, Y = self._simulator(index_arr, x_arr, Ss=s_arr)

        return task, P, Y

    # -- main scheduling loop -----------------------------------------------

    def run(
        self,
        state,
        tasks: list[Task] | None = None,
        on_tasks_complete=None,
    ):
        """Execute tasks asynchronously, updating *state* as each completes.

        All ``state.update()`` calls happen in the main (calling) thread;
        the ``ThreadPoolExecutor`` is used only to block on subprocess I/O.

        Parameters:
            state:  ``State`` instance — updated in-place with results.
            tasks:  Initial tasks to schedule (appended to any already pending).
            on_tasks_complete:
                Optional callback ``(state, completed_tasks) -> list[Task] | None``
                invoked in the main thread after processing newly completed
                tasks.  May return additional tasks to enqueue.

        Returns:
            List of ``(Task, P, Y)`` tuples for every completed task.
        """
        if tasks:
            self._pending_tasks.extend(tasks)

        all_results: list[tuple] = []

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures: dict = {}  # future -> Task

            while self._pending_tasks or futures:
                # 1. Harvest completed futures
                newly_completed: list[Task] = []
                done = [f for f in futures if f.done()]

                for future in done:
                    task = futures.pop(future)
                    try:
                        _, P, Y = future.result()
                        self._resources.release(task.cores_required)

                        # Reshape scalars / 1-D arrays for state.update
                        Y_arr = np.atleast_2d(np.asarray(Y))
                        P_arr = np.atleast_2d(np.asarray(P)) if P is not None else None
                        S_arr = np.array([[task.s]]) if task.s is not None else None

                        state.update(
                            index_next=np.array([task.index]),
                            X_next=task.x.reshape(1, -1),
                            Y_next=Y_arr,
                            P_next=P_arr,
                            S_next=S_arr,
                        )

                        newly_completed.append(task)
                        all_results.append((task, P, Y))

                        logger.info(
                            "Task %d completed | fidelity=%s | cores_released=%d | utilisation=%.0f%%",
                            task.index,
                            task.s,
                            task.cores_required,
                            self._resources.utilisation() * 100,
                        )

                    except Exception:
                        self._resources.release(task.cores_required)
                        logger.exception("Task %d failed", task.index)

                # 2. Notify caller and accept new tasks
                if newly_completed and on_tasks_complete is not None:
                    new_tasks = on_tasks_complete(state, newly_completed)
                    if new_tasks:
                        self._pending_tasks.extend(new_tasks)

                # 3. Schedule tasks that fit within available resources
                to_launch = self._schedule_next()
                for task in to_launch:
                    self._resources.allocate(task.cores_required)
                    logger.info(
                        "Launching task %d | fidelity=%s | cores=%d | utilisation=%.0f%%",
                        task.index,
                        task.s,
                        task.cores_required,
                        self._resources.utilisation() * 100,
                    )
                    future = executor.submit(self._run_single_task, task)
                    futures[future] = task

                # 4. Avoid busy-waiting
                if self._pending_tasks or futures:
                    time.sleep(self._poll_interval)

        logger.info("All tasks complete. Total: %d", len(all_results))
        return all_results
