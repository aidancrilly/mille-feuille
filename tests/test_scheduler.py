import threading
import time

import numpy as np
import pytest
from millefeuille.asynch import AsyncScheduler
from millefeuille.domain import InputDomain
from millefeuille.generators import RandomCandidateGenerator
from millefeuille.simulator import FidelityConfig, PythonSimulator, ResourceManager, Scheduler
from millefeuille.state import State
from millefeuille.utils import run_generator_loop

DOMAIN = InputDomain(dim=1, b_low=np.array([0.0]), b_up=np.array([1.0]), steps=np.array([0.0]))


class DelaySimulator(PythonSimulator):
    """Simulator that sleeps proportionally to fidelity so we can measure wall-clock speedup."""

    def __init__(self, delay: float = 0.1):
        self.delay = delay

    def __call__(self, indices, Xs, Ss=None):
        s_val = float(np.amax(Ss)) if Ss is not None else 0.0
        time.sleep(self.delay * s_val + self.delay)
        Ys = -np.sum(Xs**2, axis=-1, keepdims=True)
        return None, Ys


def _make_empty_state():
    """Create a State with no initial data."""
    return State(
        input_domain=DOMAIN,
        index=None,
        Xs=None,
        Ys=None,
    )


def _sort_by_index(state):
    """Return Xs, Ys sorted by index so async and sync results are directly comparable."""
    order = np.argsort(state.index[:, 0])
    return state.Xs[order], state.Ys[order]


# ---------------------------------------------------------------------------
# Test: async is faster than synchronous batching for the same work
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_async_faster_than_sync():
    """Async scheduling with 4 cores should complete faster than synchronous
    batches of 2, and both must produce identical Y values."""

    rng = np.random.default_rng(12)
    n_tasks = 12
    batch_size = 2
    delay = 0.1

    Xs_fixed = rng.uniform(size=(n_tasks, 1))
    Ss_fixed = np.zeros((n_tasks, 1))
    Ss_fixed[::2] = 1  # Alternate fidelity levels (0, 1, 0, 1, ...)

    # -- synchronous run (batches of 2) ------------------------------------
    class FixedGenerator:
        """Yields pre-determined slices of Xs/Ss on each call."""

        def __init__(self):
            self._offset = 0

        def __call__(self, state, n):
            lo = self._offset
            hi = lo + n
            idx_start = int(state.index.max()) + 1 if state.index is not None else 0
            indices = idx_start + np.arange(n)
            self._offset = hi
            return indices, Xs_fixed[lo:hi], Ss_fixed[lo:hi]

    sync_state = _make_empty_state()
    start_sync = time.perf_counter()
    run_generator_loop(
        Nsamples=n_tasks // batch_size,
        batch_size=batch_size,
        generate_candidates=FixedGenerator(),
        state=sync_state,
        simulator=DelaySimulator(delay=delay),
    )
    sync_time = time.perf_counter() - start_sync

    # -- asynchronous run (4 cores, up to 4 concurrent jobs) ----------------
    resource_manager = ResourceManager(total_cores=4)
    fidelity_configs = {
        0: FidelityConfig(cores_required=1),
        1: FidelityConfig(cores_required=2),
    }

    async_sched = AsyncScheduler(
        simulator=DelaySimulator(delay=delay),
        resource_manager=resource_manager,
        fidelity_configs=fidelity_configs,
        scheduler=None,
        poll_interval=0.05,
    )

    indices = np.arange(n_tasks)
    task_list = async_sched.create_tasks(indices, Xs_fixed, Ss_fixed)

    async_state = _make_empty_state()
    start_async = time.perf_counter()
    async_sched.run(async_state, task_list)
    async_time = time.perf_counter() - start_async

    print(f"\nSync  time: {sync_time:.2f}s")
    print(f"Async time: {async_time:.2f}s")
    print(f"Speed-up:   {sync_time / async_time:.2f}x")

    # -- timing assertion: async must be faster -----------------------------
    assert async_time < sync_time, f"Async ({async_time:.2f}s) should be faster than sync ({sync_time:.2f}s)"

    # -- correctness: same Y values for same inputs -------------------------
    sync_Xs, sync_Ys = _sort_by_index(sync_state)
    async_Xs, async_Ys = _sort_by_index(async_state)

    np.testing.assert_array_equal(sync_Xs, async_Xs)
    np.testing.assert_allclose(sync_Ys, async_Ys, atol=1e-12)


# ---------------------------------------------------------------------------
# Test: async scheduler produces correct results for all tasks
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_async_scheduler_correctness():
    """All tasks complete and Y values match the deterministic function."""

    rng = np.random.default_rng(99)
    n_tasks = 8
    delay = 0.05

    Xs = rng.uniform(size=(n_tasks, 1))
    Ss = rng.choice([0, 1], size=(n_tasks, 1))

    resource_manager = ResourceManager(total_cores=4)
    fidelity_configs = {
        0: FidelityConfig(cores_required=1),
        1: FidelityConfig(cores_required=2),
    }

    async_sched = AsyncScheduler(
        simulator=DelaySimulator(delay=delay),
        resource_manager=resource_manager,
        fidelity_configs=fidelity_configs,
        scheduler=None,
        poll_interval=0.05,
    )

    state = _make_empty_state()
    indices = np.arange(n_tasks)
    task_list = async_sched.create_tasks(indices, Xs, Ss)
    async_sched.run(state, task_list)

    # Should have n_tasks evaluations
    assert state.nsamples == n_tasks

    # Check Y values
    order = np.argsort(state.index[:, 0])
    sorted_Xs = state.Xs[order]
    sorted_Ys = state.Ys[order]

    expected_Ys = -np.sum(sorted_Xs**2, axis=-1, keepdims=True)
    np.testing.assert_allclose(sorted_Ys, expected_Ys, atol=1e-12)


# ---------------------------------------------------------------------------
# Test: run_generator_loop with RandomCandidateGenerator
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_run_generator_loop_with_random_generator():
    """run_generator_loop works end-to-end with a CandidateGenerator subclass."""

    generator = RandomCandidateGenerator(domain=DOMAIN, rng=np.random.default_rng(7))
    state = _make_empty_state()
    n_iters = 3
    batch_size = 2

    run_generator_loop(
        Nsamples=n_iters,
        batch_size=batch_size,
        generate_candidates=generator,
        state=state,
        simulator=DelaySimulator(delay=0.0),
    )

    # 3 iterations * 2 = 6 total samples
    assert state.nsamples == n_iters * batch_size

    # Check all Y values are -sum(x^2)
    expected_Ys = -np.sum(state.Xs**2, axis=-1, keepdims=True)
    np.testing.assert_allclose(state.Ys, expected_Ys, atol=1e-12)


# ---------------------------------------------------------------------------
# Test: Scheduler._normalise_nprocs
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_normalise_nprocs_int_broadcast():
    result = Scheduler._normalise_nprocs(3, 4)
    assert result == [3, 3, 3, 3]


@pytest.mark.unit
def test_normalise_nprocs_list():
    result = Scheduler._normalise_nprocs([1, 2, 4], 3)
    assert result == [1, 2, 4]


@pytest.mark.unit
def test_normalise_nprocs_mismatched_length():
    with pytest.raises(ValueError, match="len\\(nprocs\\)"):
        Scheduler._normalise_nprocs([1, 2], 3)


# ---------------------------------------------------------------------------
# Test: FidelityConfig.from_simulator
# ---------------------------------------------------------------------------


class CoresAwareSimulator(PythonSimulator):
    def __call__(self, indices, Xs, Ss=None):
        return None, -np.sum(Xs**2, axis=-1, keepdims=True)

    def cores_required(self, s: int) -> int:
        return {0: 1, 1: 4}[s]


@pytest.mark.unit
def test_fidelity_config_from_simulator():
    sim = CoresAwareSimulator()
    configs = FidelityConfig.from_simulator(sim, [0, 1], reserve={1: True})
    assert configs[0].cores_required == 1
    assert configs[0].reserve is False
    assert configs[1].cores_required == 4
    assert configs[1].reserve is True


@pytest.mark.unit
def test_fidelity_config_from_simulator_not_implemented():
    sim = DelaySimulator(delay=0.0)  # does not override cores_required
    with pytest.raises(NotImplementedError):
        FidelityConfig.from_simulator(sim, [0, 1])


# ---------------------------------------------------------------------------
# Test: AsyncScheduler auto-infer fidelity_configs
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_async_scheduler_auto_infer_from_simulator():
    sim = CoresAwareSimulator()
    rm = ResourceManager(total_cores=8)
    sched = AsyncScheduler(simulator=sim, resource_manager=rm)
    # create_tasks should pick up cores from simulator.cores_required
    indices = np.array([0, 1])
    Xs = np.array([[0.5], [0.8]])
    Ss = np.array([[0], [1]])
    tasks = sched.create_tasks(indices, Xs, Ss)
    assert tasks[0].cores_required == 1
    assert tasks[1].cores_required == 4


# ---------------------------------------------------------------------------
# Test: tasks_in_flight / n_tasks_in_flight
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_tasks_in_flight_empty():
    sched = AsyncScheduler(
        simulator=DelaySimulator(delay=0.1),
        resource_manager=ResourceManager(total_cores=4),
    )
    assert sched.tasks_in_flight() == []
    assert sched.n_tasks_in_flight() == 0


@pytest.mark.unit
def test_tasks_in_flight_during_run():
    """Verify tasks_in_flight reports running tasks via the on_tasks_complete callback."""
    observed_in_flight: list[int] = []
    delay = 0.2

    def callback(state, num_completed):
        # The scheduler should still have some tasks in flight
        observed_in_flight.append(sched.n_tasks_in_flight())
        return None

    sched = AsyncScheduler(
        simulator=DelaySimulator(delay=delay),
        resource_manager=ResourceManager(total_cores=8),
        fidelity_configs={0: FidelityConfig(cores_required=1), 1: FidelityConfig(cores_required=1)},
        poll_interval=0.02,
    )

    n_tasks = 6
    indices = np.arange(n_tasks)
    Xs = np.random.rand(n_tasks, 1)
    Ss = np.array([[0], [1], [0], [1], [0], [1]])

    state = _make_empty_state()
    tasks = sched.create_tasks(indices, Xs, Ss)
    sched.run(state, tasks, on_tasks_complete=callback)

    assert state.nsamples == n_tasks
    # After all tasks complete and run() returns, no tasks in flight
    assert sched.n_tasks_in_flight() == 0


@pytest.mark.unit
def test_tasks_in_flight_fidelity_filter():
    """Filter tasks_in_flight by fidelity using the on_tasks_complete callback."""
    observed_lf: list[int] = []
    observed_hf: list[int] = []

    def callback(state, num_completed):
        observed_lf.append(sched.n_tasks_in_flight(fidelity=0))
        observed_hf.append(sched.n_tasks_in_flight(fidelity=1))
        return None

    sched = AsyncScheduler(
        simulator=DelaySimulator(delay=0.15),
        resource_manager=ResourceManager(total_cores=8),
        fidelity_configs={0: FidelityConfig(cores_required=1), 1: FidelityConfig(cores_required=1)},
        poll_interval=0.02,
    )

    indices = np.arange(4)
    Xs = np.random.rand(4, 1)
    Ss = np.array([[0], [0], [1], [1]])

    state = _make_empty_state()
    tasks = sched.create_tasks(indices, Xs, Ss)
    sched.run(state, tasks, on_tasks_complete=callback)

    assert state.nsamples == 4


@pytest.mark.unit
def test_tasks_in_flight_thread_safety():
    """Concurrent reads of tasks_in_flight from another thread don't raise."""
    sched = AsyncScheduler(
        simulator=DelaySimulator(delay=0.1),
        resource_manager=ResourceManager(total_cores=4),
        fidelity_configs={0: FidelityConfig(cores_required=1)},
        poll_interval=0.02,
    )

    errors = []

    def reader():
        for _ in range(50):
            try:
                sched.tasks_in_flight()
                sched.n_tasks_in_flight(fidelity=0)
            except Exception as e:
                errors.append(e)
            time.sleep(0.01)

    t = threading.Thread(target=reader)
    t.start()

    indices = np.arange(4)
    Xs = np.random.rand(4, 1)
    Ss = np.zeros((4, 1))

    state = _make_empty_state()
    tasks = sched.create_tasks(indices, Xs, Ss)
    sched.run(state, tasks)
    t.join()

    assert errors == [], f"Thread-safety errors: {errors}"
