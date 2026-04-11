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
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import numpy.typing as npt

from .generators import CandidateGenerator
from .simulator import *

logger = logging.getLogger("millefeuille.scheduler")


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
                Optional callback 
                ``(state, completed_tasks) -> list[Task] | None, state``
                invoked in the main thread after processing newly completed
                tasks.  May return additional tasks to enqueue. 
                May update state from disk

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
                num_completed = 0
                done = [f for f in futures if f.done()]

                for future in done:
                    task = futures.pop(future)
                    num_completed += 1
                    try:
                        _, P, Y = future.result()
                        self._resources.release(task.cores_required)

                        # Reshape scalars / 1-D arrays for state.update
                        Y_arr = np.asarray(Y).reshape(1, -1)
                        P_arr = np.asarray(P).reshape(1, -1) if P is not None else None
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
                if num_completed > 0 and on_tasks_complete is not None:
                    new_tasks, state = on_tasks_complete(state, num_completed)
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


def run_async_loop(
    total_evaluations,
    generate_candidates,
    state,
    simulator,
    resource_manager,
    scheduler=None,
    fidelity_configs=None,
    refill_interval=None,
    index_start=None,
    max_workers=16,
    poll_interval=0.5,
    db_name=None,
    verbose=False,
):
    """Generic asynchronous evaluation loop with a pluggable candidate generator.

    Unlike ``run_async_Bayesian_optimiser`` (which is tied to surrogate-based
    acquisition), this function accepts **any** callable that produces new
    candidates.  It is suitable for random sampling, threshold sampling,
    adaptive strategies, or custom heuristics.

    Parameters:
        total_evaluations:
            Total number of simulation evaluations to perform.
        generate_candidates:
            A ``CandidateGenerator`` instance or a callable with signature::

                generate_candidates(state, budget) -> (Xs, Ss | None)

            * *state* — current ``State`` (read-only is fine).
            * *budget* — how many new candidates are requested.
            * Returns ``Xs`` of shape ``(N, dim)`` and optionally ``Ss``
              of shape ``(N, 1)`` (or ``None`` for single-fidelity).
              The function may return fewer than *budget* candidates.

        state:              Current ``State``.
        simulator:          ``ExectuableSimulator`` or ``PythonSimulator``.
        resource_manager:   ``ResourceManager`` tracking available cores.
        scheduler:          ``Scheduler`` instance (required for
                            ``ExectuableSimulator``).
        fidelity_configs:   Optional ``{fidelity: FidelityConfig}`` mapping.
        refill_interval:    Request new candidates every *N* completions
                            (default: first batch size).
        max_workers:        Thread-pool size (default 16).
        poll_interval:      Seconds between scheduling checks (default 0.5).
        db_name:            Optional file path to persist state.  Uses
                            ``to_csv`` for ``.csv`` extensions and ``save``
                            (SQLite) for anything else (e.g. ``.db``).
        verbose:            Enable info-level log messages.

    Returns:
        Updated ``State``.
    """

    def _call_generator(generate_candidates, state, budget):
        """Invoke a candidate generator, handling both ``CandidateGenerator`` instances and plain callables.

        Returns:
            (Xs, Ss) where Ss may be ``None``.
        """
        if isinstance(generate_candidates, CandidateGenerator):
            return generate_candidates.generate(state, budget)
        result = generate_candidates(state, budget)
        if isinstance(result, tuple) and len(result) == 2:
            return result
        return result, None

    if isinstance(simulator, ExectuableSimulator) and scheduler is None:
        raise ValueError("If simulator is an ExectuableSimulator, you must provide a scheduler")

    async_sched = AsyncScheduler(
        simulator=simulator,
        resource_manager=resource_manager,
        scheduler=scheduler,
        fidelity_configs=fidelity_configs,
        max_workers=max_workers,
        poll_interval=poll_interval,
    )

    # --- initial candidates ------------------------------------------------
    initial_budget = min(total_evaluations, resource_manager.total)
    X_init, S_init = _call_generator(generate_candidates, state, initial_budget)

    n_init = X_init.shape[0]
    if n_init == 0:
        raise ValueError(
            "The initial candidate generator returned an empty batch; "
            "run_async_loop requires at least one initial candidate."
        )
    if index_start is None:
        index_start = int(state.index.max()) + 1 if state.index is not None else 0
    idx_init = index_start + np.arange(n_init)
    initial_tasks = async_sched.create_tasks(idx_init, X_init, S_init)

    if refill_interval is None:
        refill_interval = max(1, n_init)

    # --- book-keeping ------------------------------------------------------
    evaluations_launched = [n_init]
    completions_since_refill = [0]
    idx_next = [idx_init[-1]]

    def _on_tasks_complete(state, num_completed_tasks):
        completions_since_refill[0] += num_completed_tasks

        if db_name is not None:
            _ext = os.path.splitext(db_name)[1].lower()
            if _ext == ".csv":
                state.to_csv(db_name)
            else:
                from .state import State

                state.save(db_name)
                state = State.load(db_name, Y_scaler=state.Y_scaler)

        remaining = total_evaluations - evaluations_launched[0]
        if remaining <= 0:
            return None

        if completions_since_refill[0] >= refill_interval:
            completions_since_refill[0] = 0
            budget = min(refill_interval, remaining)

            if verbose:
                logger.info(
                    "Generating new candidates (launched=%d/%d)",
                    evaluations_launched[0],
                    total_evaluations,
                )

            X_new, S_new = _call_generator(generate_candidates, state, budget)

            n_new = X_new.shape[0]
            if n_new == 0:
                raise ValueError("generator returned 0 candidates inside run_async_loop...")
            idx_start = idx_next[0] + 1
            idx_new = idx_start + np.arange(n_new)
            new_tasks = async_sched.create_tasks(idx_new, X_new, S_new)

            idx_next[0] = idx_new[-1]
            evaluations_launched[0] += n_new
            return new_tasks, state

        return None

    # --- run ---------------------------------------------------------------
    async_sched.run(state, initial_tasks, on_tasks_complete=_on_tasks_complete)

    return state
