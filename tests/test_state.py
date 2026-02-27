import csv
import os
import tempfile

import numpy as np
import pytest
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
