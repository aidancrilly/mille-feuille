import time

import numpy as np
import pytest
from millefeuille.asynch import AsyncScheduler
from millefeuille.domain import InputDomain
from millefeuille.generators import RandomCandidateGenerator
from millefeuille.simulator import FidelityConfig, PythonSimulator, ResourceManager
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
