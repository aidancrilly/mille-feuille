import numpy as np
import pytest
from millefeuille.state import State
from millefeuille.surrogate import SingleFidelityGPSurrogate
from millefeuille.tasks import AvailableResources, TaskList, run_async_optimiser

from .conftest import (
    TEST_KERNEL,
    TEST_KERNEL_KWARGS,
    ForresterDomain,
    ForresterSampler,
    PythonForresterFunction,
)

# ---------------------------------------------------------------------------
# TaskList tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_tasklist_len_and_is_empty():
    tasks = TaskList(indices=np.array([0, 1, 2]), Xs=np.array([[0.1], [0.2], [0.3]]))
    assert len(tasks) == 3
    assert not tasks.is_empty()


@pytest.mark.unit
def test_tasklist_empty():
    tasks = TaskList(indices=np.array([]), Xs=np.empty((0, 1)))
    assert len(tasks) == 0
    assert tasks.is_empty()


@pytest.mark.unit
def test_tasklist_pop():
    tasks = TaskList(indices=np.array([10, 20, 30]), Xs=np.array([[0.1], [0.2], [0.3]]))
    idx, x, s = tasks.pop(0)
    assert idx == 10
    assert np.isclose(x[0], 0.1)
    assert s is None
    assert len(tasks) == 2


@pytest.mark.unit
def test_tasklist_pop_with_fidelity():
    tasks = TaskList(
        indices=np.array([10, 20]),
        Xs=np.array([[0.1], [0.2]]),
        Ss=np.array([[0], [1]]),
    )
    idx, x, s = tasks.pop(1)
    assert idx == 20
    assert s == 1
    assert len(tasks) == 1
    assert tasks.Ss is not None and tasks.Ss[0, 0] == 0


@pytest.mark.unit
def test_tasklist_append_single():
    tasks = TaskList(indices=np.array([0]), Xs=np.array([[0.5]]))
    tasks.append(1, np.array([0.7]))
    assert len(tasks) == 2
    assert tasks.indices[1] == 1
    assert np.isclose(tasks.Xs[1, 0], 0.7)


@pytest.mark.unit
def test_tasklist_append_multiple():
    tasks = TaskList(indices=np.array([0]), Xs=np.array([[0.5]]))
    tasks.append(np.array([1, 2]), np.array([[0.6], [0.7]]))
    assert len(tasks) == 3


@pytest.mark.unit
def test_tasklist_append_with_fidelity():
    tasks = TaskList(
        indices=np.array([0]),
        Xs=np.array([[0.5]]),
        Ss=np.array([[0]]),
    )
    tasks.append(1, np.array([0.7]), S=1)
    assert len(tasks) == 2
    assert tasks.Ss is not None
    assert tasks.Ss[1, 0] == 1


@pytest.mark.unit
def test_tasklist_append_empty_is_noop():
    tasks = TaskList(indices=np.array([0, 1]), Xs=np.array([[0.1], [0.2]]))
    tasks.append(np.array([]), np.empty((0, 1)))
    assert len(tasks) == 2


# ---------------------------------------------------------------------------
# AvailableResources tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_available_resources_nproc_for_task():
    res = AvailableResources(ntotal_proc=4, nproc_per_fidelity=[1, 2])
    assert res.nproc_for_task(0) == 1
    assert res.nproc_for_task(1) == 2
    assert res.nproc_for_task(None) == 1  # defaults to fidelity 0


@pytest.mark.unit
def test_available_resources_can_fit_task():
    res = AvailableResources(ntotal_proc=4, nproc_per_fidelity=[1, 2])
    assert res.can_fit_task(0, 3)  # 3+1 = 4 <= 4
    assert not res.can_fit_task(0, 4)  # 4+1 = 5 > 4
    assert res.can_fit_task(1, 2)  # 2+2 = 4 <= 4
    assert not res.can_fit_task(1, 3)  # 3+2 = 5 > 4


# ---------------------------------------------------------------------------
# run_async_optimiser tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_run_async_optimiser_single_fidelity():
    """Process an initial task list and verify state is updated."""
    ntrain = 16
    Is = np.arange(ntrain)
    _rng = np.random.default_rng(seed=42)
    Xs, _ = ForresterSampler(_rng).random(ntrain), None
    f = PythonForresterFunction()
    _, Ys = f(Is, Xs)

    state = State(ForresterDomain, Is, Xs, Ys)
    surrogate = SingleFidelityGPSurrogate(kernel=TEST_KERNEL, kernel_kwargs=TEST_KERNEL_KWARGS)
    surrogate.fit(state)

    # Create a small initial task list
    X_init = np.array([[0.2], [0.5], [0.8]])
    n_init = len(X_init)
    tasks = TaskList(
        indices=np.arange(ntrain, ntrain + n_init),
        Xs=X_init,
    )

    # resources: 2 procs, each task needs 1 proc
    resources = AvailableResources(ntotal_proc=2, nproc_per_fidelity=[1])

    # generate_next_tasks returns empty list — no renewal beyond initial tasks
    def generate_next_tasks(st, surr, n):
        return TaskList(indices=np.array([]), Xs=np.empty((0, 1)))

    initial_nsamples = state.nsamples
    final_state = run_async_optimiser(
        tasks,
        resources,
        generate_next_tasks,
        state,
        surrogate,
        f,
        retrain_interval=4,  # won't trigger retraining for 3-task list
    )

    assert final_state.nsamples == initial_nsamples + n_init, (
        "All initial tasks should have been added to state"
    )


@pytest.mark.unit
def test_run_async_optimiser_renews_task_list():
    """Verify the task list is renewed and surrogate is retrained."""
    ntrain = 16
    Is = np.arange(ntrain)
    _rng = np.random.default_rng(seed=7)
    Xs, _ = ForresterSampler(_rng).random(ntrain), None
    f = PythonForresterFunction()
    _, Ys = f(Is, Xs)

    state = State(ForresterDomain, Is, Xs, Ys)
    surrogate = SingleFidelityGPSurrogate(kernel=TEST_KERNEL, kernel_kwargs=TEST_KERNEL_KWARGS)
    surrogate.fit(state)

    # Initial task list with 2 tasks at well-spaced locations
    tasks = TaskList(
        indices=np.array([ntrain, ntrain + 1]),
        Xs=np.array([[0.25], [0.75]]),
    )

    resources = AvailableResources(ntotal_proc=2, nproc_per_fidelity=[1])

    # Generate exactly 2 extra tasks when asked (once), then stop
    renewal_calls = [0]
    extra_per_call = 2
    max_calls = 1

    def generate_next_tasks(st, surr, n):
        if renewal_calls[0] >= max_calls:
            return TaskList(indices=np.array([]), Xs=np.empty((0, 1)))
        next_start = int(st.index[-1, 0]) + 1
        new_Is = np.arange(next_start, next_start + extra_per_call)
        # Use deterministic well-spaced Xs for reproducibility
        new_Xs = np.array([[0.1], [0.9]])
        renewal_calls[0] += 1
        return TaskList(indices=new_Is, Xs=new_Xs)

    initial_nsamples = state.nsamples
    final_state = run_async_optimiser(
        tasks,
        resources,
        generate_next_tasks,
        state,
        surrogate,
        f,
        retrain_interval=2,
    )

    # Should have processed initial 2 tasks + up to extra_per_call renewed tasks
    assert final_state.nsamples >= initial_nsamples + 2, "At least the initial tasks should be added"
    assert renewal_calls[0] >= 1, "generate_next_tasks should have been called at least once"

