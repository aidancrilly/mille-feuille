import csv
import os
import sqlite3
import tempfile

import numpy as np
import pytest
from millefeuille.domain import InputDomain
from millefeuille.state import State

from .conftest import ForresterDomain


def make_simple_state(n=5, index_start=0):
    Is = np.arange(index_start, index_start + n, dtype=float)
    Xs = np.random.rand(n, 1)
    Ys = np.random.rand(n, 1)
    return State(ForresterDomain, Is, Xs, Ys)


@pytest.mark.unit
def test_to_csv_creates_file():
    state = make_simple_state(n=5)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        fname = tmp.name
    try:
        os.remove(fname)
        state.to_csv(fname)
        assert os.path.exists(fname)
        with open(fname, "r") as f:
            rows = list(csv.reader(f))
        # header + 5 data rows
        assert len(rows) == 6
    finally:
        if os.path.exists(fname):
            os.remove(fname)


@pytest.mark.unit
def test_to_csv_no_duplicate_index_on_second_call():
    """Calling to_csv twice with the same state should not duplicate rows."""
    state = make_simple_state(n=5)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        fname = tmp.name
    try:
        os.remove(fname)
        state.to_csv(fname)
        state.to_csv(fname)  # second call with same data
        with open(fname, "r") as f:
            rows = list(csv.reader(f))
        # header + 5 data rows (no duplicates)
        assert len(rows) == 6
    finally:
        if os.path.exists(fname):
            os.remove(fname)


@pytest.mark.unit
def test_to_csv_appends_new_index_rows():
    """After updating state with new indices, to_csv should append only new rows."""
    state = make_simple_state(n=5, index_start=0)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        fname = tmp.name
    try:
        os.remove(fname)
        state.to_csv(fname)

        # Update state with 3 new samples (indices 5, 6, 7)
        new_Is = np.array([5.0, 6.0, 7.0])
        new_Xs = np.random.rand(3, 1)
        new_Ys = np.random.rand(3, 1)
        state.update(new_Is, new_Xs, new_Ys)
        state.to_csv(fname)

        with open(fname, "r") as f:
            rows = list(csv.reader(f))
        # header + 5 original + 3 new = 9 rows total
        assert len(rows) == 9
    finally:
        if os.path.exists(fname):
            os.remove(fname)


@pytest.mark.unit
def test_to_csv_skips_existing_index_values():
    """Rows with index values already in the CSV are not overwritten."""
    state = make_simple_state(n=5, index_start=0)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        fname = tmp.name
    try:
        os.remove(fname)
        state.to_csv(fname)

        # Read first written Y value
        with open(fname, "r") as f:
            rows = list(csv.reader(f))
        first_y_col = rows[0].index(state.Y_names[0])
        original_y = rows[1][first_y_col]

        # Update state with one overlapping index (0) and one new index (5)
        new_Is = np.array([0.0, 5.0])
        new_Xs = np.random.rand(2, 1)
        new_Ys = np.random.rand(2, 1)
        state.update(new_Is, new_Xs, new_Ys)
        state.to_csv(fname)

        with open(fname, "r") as f:
            rows = list(csv.reader(f))

        # Should have header + 5 original + 1 new (index 5 only) = 7 rows
        assert len(rows) == 7

        # The original row for index 0 should be unchanged
        assert rows[1][first_y_col] == original_y
    finally:
        if os.path.exists(fname):
            os.remove(fname)


@pytest.mark.unit
def test_filling_empty_state():
    dummy_domain = InputDomain(dim=1, b_low=np.array([0.0]), b_up=np.array([1.0]), steps=np.zeros(1))
    empty_state = State(dummy_domain, index=None, Xs=None, Ys=None)

    assert empty_state.index is None
    assert empty_state.Xs is None
    assert empty_state.Ys is None
    assert empty_state.Ps is None
    assert empty_state.Ss is None

    assert empty_state.best_value == -np.inf
    assert empty_state.best_value_transformed == -np.inf

    # Add some data and check best value updates
    new_Is = np.array([0.0, 1.0])
    new_Xs = np.array([[0.5], [0.8]])
    new_Ys = np.array([[0.3], [0.9]])
    empty_state.update(new_Is, new_Xs, new_Ys)

    assert empty_state.best_value == 0.9
    assert empty_state.best_value_transformed > -np.inf

    # Add single entry
    empty_state = State(dummy_domain, index=None, Xs=None, Ys=None)

    new_Is = np.array([0.0])
    new_Xs = np.array([[0.5]])
    new_Ys = np.array([[0.3]])
    empty_state.update(new_Is, new_Xs, new_Ys)

    assert empty_state.best_value == 0.3
    assert empty_state.best_value_transformed > -np.inf


# ---------------------------------------------------------------------------
# SQLite persistence tests  (save / load)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_save_creates_db():
    """save() creates a new SQLite database with the correct row count."""
    state = make_simple_state(n=5)
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        fname = tmp.name
    try:
        os.remove(fname)
        state.save(fname)
        assert os.path.exists(fname)

        conn = sqlite3.connect(fname)
        rows = conn.execute("SELECT COUNT(*) FROM state").fetchone()[0]
        conn.close()
        assert rows == 5
    finally:
        if os.path.exists(fname):
            os.remove(fname)


@pytest.mark.unit
def test_save_no_duplicate_on_second_call():
    """Calling save() twice with the same state should not duplicate rows."""
    state = make_simple_state(n=5)
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        fname = tmp.name
    try:
        os.remove(fname)
        state.save(fname)
        state.save(fname)

        conn = sqlite3.connect(fname)
        rows = conn.execute("SELECT COUNT(*) FROM state").fetchone()[0]
        conn.close()
        assert rows == 5
    finally:
        if os.path.exists(fname):
            os.remove(fname)


@pytest.mark.unit
def test_save_appends_new_rows():
    """After updating state with new indices, save() appends only new rows."""
    state = make_simple_state(n=5, index_start=0)
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        fname = tmp.name
    try:
        os.remove(fname)
        state.save(fname)

        new_Is = np.array([5.0, 6.0, 7.0])
        new_Xs = np.random.rand(3, 1)
        new_Ys = np.random.rand(3, 1)
        state.update(new_Is, new_Xs, new_Ys)
        state.save(fname)

        conn = sqlite3.connect(fname)
        rows = conn.execute("SELECT COUNT(*) FROM state").fetchone()[0]
        conn.close()
        assert rows == 8  # 5 original + 3 new
    finally:
        if os.path.exists(fname):
            os.remove(fname)


@pytest.mark.unit
def test_save_skips_existing_index():
    """Rows with index values already in the DB are not overwritten."""
    state = make_simple_state(n=5, index_start=0)
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        fname = tmp.name
    try:
        os.remove(fname)
        state.save(fname)

        # Record the original Y value for index 0
        conn = sqlite3.connect(fname)
        original_y = conn.execute(
            f'SELECT "{state.Y_names[0]}" FROM state WHERE "{state.index_names[0]}" = 0.0'
        ).fetchone()[0]
        conn.close()

        # Update with overlapping index (0) and new index (5)
        new_Is = np.array([0.0, 5.0])
        new_Xs = np.random.rand(2, 1)
        new_Ys = np.random.rand(2, 1)
        state.update(new_Is, new_Xs, new_Ys)
        state.save(fname)

        conn = sqlite3.connect(fname)
        rows = conn.execute("SELECT COUNT(*) FROM state").fetchone()[0]
        saved_y = conn.execute(
            f'SELECT "{state.Y_names[0]}" FROM state WHERE "{state.index_names[0]}" = 0.0'
        ).fetchone()[0]
        conn.close()

        assert rows == 6  # 5 original + 1 new (index 5)
        assert saved_y == original_y  # Original row unchanged
    finally:
        if os.path.exists(fname):
            os.remove(fname)


@pytest.mark.unit
def test_save_load_roundtrip():
    """Data saved and loaded back should match."""
    state = make_simple_state(n=5)
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        fname = tmp.name
    try:
        os.remove(fname)
        state.save(fname)
        loaded = State.load(fname)

        np.testing.assert_array_almost_equal(loaded.index, state.index)
        np.testing.assert_array_almost_equal(loaded.Xs, state.Xs)
        np.testing.assert_array_almost_equal(loaded.Ys, state.Ys)
        assert loaded.index_names == state.index_names
        assert loaded.X_names == state.X_names
        assert loaded.Y_names == state.Y_names
    finally:
        if os.path.exists(fname):
            os.remove(fname)


@pytest.mark.unit
def test_save_raises_on_missing_data():
    """save() should raise ValueError when required arrays are missing."""
    dummy_domain = InputDomain(dim=1, b_low=np.array([0.0]), b_up=np.array([1.0]), steps=np.zeros(1))
    empty_state = State(dummy_domain, index=None, Xs=None, Ys=None)
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        fname = tmp.name
    try:
        with pytest.raises(ValueError):
            empty_state.save(fname)
    finally:
        if os.path.exists(fname):
            os.remove(fname)


@pytest.mark.unit
def test_save_load_with_parameters():
    """Roundtrip with Ps (parameters) present."""
    Is = np.arange(5, dtype=float)
    Xs = np.random.rand(5, 2)
    Ys = np.random.rand(5, 1)
    Ps = np.random.rand(5, 3)
    domain = InputDomain(dim=2, b_low=np.zeros(2), b_up=np.ones(2), steps=np.zeros(2))
    state = State(domain, Is, Xs, Ys, Ps=Ps, P_names=["p_a", "p_b", "p_c"])
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        fname = tmp.name
    try:
        os.remove(fname)
        state.save(fname)
        loaded = State.load(fname)

        np.testing.assert_array_almost_equal(loaded.Ps, state.Ps)
        assert loaded.P_names == state.P_names
    finally:
        if os.path.exists(fname):
            os.remove(fname)


@pytest.mark.unit
def test_save_load_multi_column_index():
    """Roundtrip with multi-column composite index."""
    n = 5
    Is = np.column_stack([np.arange(n, dtype=float), np.arange(10, 10 + n, dtype=float)])
    Xs = np.random.rand(n, 1)
    Ys = np.random.rand(n, 1)
    state = State(ForresterDomain, Is, Xs, Ys, index_names=["batch", "sample"])

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        fname = tmp.name
    try:
        os.remove(fname)
        state.save(fname)
        loaded = State.load(fname)

        np.testing.assert_array_almost_equal(loaded.index, state.index)
        assert loaded.index_names == ["batch", "sample"]
    finally:
        if os.path.exists(fname):
            os.remove(fname)
