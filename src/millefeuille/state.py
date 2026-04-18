import csv
import json
import logging
import os
import sqlite3
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
from botorch.models.transforms.outcome import Standardize

from .definitions import device, dtype
from .domain import FidelityDomain, InputDomain, ScaleFactorInputDomain

"""
Defines the optimiser state
"""


def check_for_2D_shape(arrays):
    # Ensure correct shape from each non-None array
    cleaned_arrays = []
    for arr in arrays:
        if arr is not None:
            if arr.ndim == 1:
                cleaned_arrays.append(arr.reshape(-1, 1))
            elif arr.ndim == 2:
                cleaned_arrays.append(arr)
            else:
                print("WARNING: State recieved non-1 or -2D array!")
                cleaned_arrays.append(arr)
        else:
            cleaned_arrays.append(None)

    return cleaned_arrays


def remove_nan_rows(arrays):
    """
    Removes rows (axis=0) where any non-None array has NaNs.

    Parameters:
        arrays (list): A list of 2D NumPy arrays and/or None values.

    Returns:
        list: A list of the same structure with rows containing NaNs removed.
    """
    # Identify non-None arrays
    valid_arrays = [arr for arr in arrays if arr is not None]

    if len(valid_arrays) == 0:
        return arrays

    # Find indices with NaNs across any of the valid arrays
    nan_mask = np.zeros(valid_arrays[0].shape[0], dtype=bool)
    for arr in valid_arrays:
        nan_mask |= np.isnan(arr).any(axis=1)

    # Remove rows at indices with NaNs from each non-None array
    cleaned_arrays = [arr[~nan_mask] if arr is not None else None for arr in arrays]
    return cleaned_arrays


def _numpy_dtype_to_sqlite(dt):
    """Map a numpy dtype to a SQLite column type string."""
    if np.issubdtype(dt, np.integer):
        return "INTEGER"
    return "REAL"


def _quote_id(name: str) -> str:
    """Double-quote a SQL identifier, escaping embedded double-quotes."""
    return '"' + name.replace('"', '""') + '"'


class StandardScaler(Standardize):
    def __init__(self, m=1):
        super().__init__(m=m)
        self.training_override = False

    def fit(self, Y):
        Y_torch = torch.tensor(Y, dtype=dtype, device=device)
        self.train()
        _ = self.forward(Y_torch)
        self.eval()

    def _tensor_check(self, Y):
        if torch.is_tensor(Y):
            return Y
        else:
            return torch.tensor(Y, dtype=dtype, device=device)

    def transform(self, Y, return_torch=True):
        Y_torch = self._tensor_check(Y)

        if self.training_override:
            self.eval()

        if return_torch:
            # Get the transformed values without transformed noise
            Y_return = self.forward(Y_torch)[0]
        else:
            # Get the transformed values without transformed noise
            Y_torch = self.forward(Y_torch)[0]
            Y_return = Y_torch.detach().cpu().numpy()

        if self.training_override:
            self.train()

        return Y_return

    def inverse_transform(self, Y, Ystd, return_torch=True):
        Y_torch = self._tensor_check(Y)
        Yvar_torch = self._tensor_check(Ystd**2)  # Convert std to var

        if return_torch:
            return self.untransform(Y_torch, Yvar_torch)
        else:
            Y_torch, Yvar_torch = self.untransform(Y_torch, Yvar_torch)
            return Y_torch.detach().cpu().numpy(), np.sqrt(Yvar_torch.detach().cpu().numpy())  # Convert var back to std


@dataclass
class State:
    """
    State containing information on problem and its progress

    input_domain = input space InputDomain
    fidelity_domain = fidelity space FidelityDomain

    index = each sample is given an index (doesn't necessarily just count the samples)
    Xs = input space samples
    Ys = cost function values
    Ps = additional scalar values to store
    Ss = model fidelities of samples

    Y_scaler = transformation on output space for training, e.g. standardise

    """

    input_domain: type[InputDomain]
    index: None | npt.NDArray
    Xs: None | npt.NDArray
    Ys: None | npt.NDArray
    Ps: None | npt.NDArray = None
    Ss: None | npt.NDArray = None

    index_names: None | list[str] = None
    X_names: None | list[str] = None
    Y_names: None | list[str] = None
    P_names: None | list[str] = None
    S_names: None | list[str] = None

    Y_scaler: None | object = None

    fidelity_domain: None | type[FidelityDomain] = None

    nsamples: int = 0
    best_value: float = -float("inf")
    best_value_transformed: float = -float("inf")

    def __post_init__(self):
        self.dim = self.input_domain.dim

        if self.Y_scaler is None:
            self.Y_scaler = StandardScaler()

        if self.fidelity_domain is not None:
            self.fidelity_domain.combine_with_input_domain(self.dim)
            self.target_fidelity = self.fidelity_domain.target_fidelity
            self.fidelity_features = self.fidelity_domain.fidelity_features
            self.l_MultiFidelity = True
        else:
            self.l_MultiFidelity = False

        # Check for 1D arrays
        self.index, self.Xs, self.Ys, self.Ps, self.Ss = check_for_2D_shape(
            [self.index, self.Xs, self.Ys, self.Ps, self.Ss]
        )

        # Remove NaN-ed indices
        self.index, self.Xs, self.Ys, self.Ps, self.Ss = remove_nan_rows(
            [self.index, self.Xs, self.Ys, self.Ps, self.Ss]
        )

        # Check name lengths against arrays
        if self.index is not None:
            self.index_names = self.auto_naming_and_check(self.index_names, self.index.shape, default_prefix="index_")
        if self.Xs is not None:
            self.X_names = self.auto_naming_and_check(self.X_names, self.Xs.shape, default_prefix="x_")
        if self.Ys is not None:
            self.Y_names = self.auto_naming_and_check(self.Y_names, self.Ys.shape, default_prefix="y_")
        if self.Ps is not None:
            self.P_names = self.auto_naming_and_check(self.P_names, self.Ps.shape, default_prefix="p_")
        if self.Ss is not None:
            self.S_names = self.auto_naming_and_check(self.S_names, self.Ss.shape, default_prefix="s_")

        if self.Ys is not None:
            self.Y_scaler.fit(self.Ys)
            if self.l_MultiFidelity:
                Ys_target = self.Ys[self.Ss[:, 0] == self.target_fidelity, :]
            else:
                Ys_target = self.Ys.copy()
            self.best_value = Ys_target.max(axis=0)
            self.best_value_transformed = self.Y_scaler.transform(self.best_value, return_torch=False)
            self.nsamples = self.Ys.shape[0]

    def filter(self, mask: npt.NDArray) -> "State":
        """Return a new State containing only rows where *mask* is True.

        Parameters
        ----------
        mask : np.ndarray
            Boolean array of length ``nsamples``.

        Returns
        -------
        State
            Filtered copy preserving all columns including ``Ss``.
        """
        if not mask.any():
            return State(
                input_domain=self.input_domain,
                index=None,
                Xs=None,
                Ys=None,
                Ps=None,
                Ss=None,
                index_names=self.index_names,
                X_names=self.X_names,
                Y_names=self.Y_names,
                P_names=self.P_names,
                S_names=self.S_names,
                Y_scaler=None,
                fidelity_domain=self.fidelity_domain,
            )

        return State(
            input_domain=self.input_domain,
            index=self.index[mask] if self.index is not None else None,
            Xs=self.Xs[mask] if self.Xs is not None else None,
            Ys=self.Ys[mask] if self.Ys is not None else None,
            Ps=self.Ps[mask] if self.Ps is not None else None,
            Ss=self.Ss[mask] if self.Ss is not None else None,
            index_names=self.index_names,
            X_names=self.X_names,
            Y_names=self.Y_names,
            P_names=self.P_names,
            S_names=self.S_names,
            Y_scaler=None,
            fidelity_domain=self.fidelity_domain,
        )

    def filter_by_fidelity(self, fidelity_value: int) -> "State":
        """Return a new single-fidelity State for the given fidelity level.

        Parameters
        ----------
        fidelity_value : int
            The fidelity level to select (compared against ``Ss[:, 0]``).

        Returns
        -------
        State
            Filtered State **without** ``Ss``, ``S_names``, or
            ``fidelity_domain`` (it represents a single-fidelity slice).

        Raises
        ------
        ValueError
            If ``self.Ss`` is ``None``.
        """
        if self.Ss is None:
            raise ValueError("State has no fidelity column (Ss)")

        mask = self.Ss[:, 0] == fidelity_value
        filtered = self.filter(mask)

        # Strip fidelity information from the single-fidelity slice
        filtered.Ss = None
        filtered.S_names = None
        filtered.fidelity_domain = None
        filtered.l_MultiFidelity = False

        return filtered

    def update(self, index_next, X_next, Y_next, S_next=None, P_next=None, refit_scaler=True):
        # Check for 1D arrays
        index_next, X_next, Y_next, P_next, S_next = check_for_2D_shape([index_next, X_next, Y_next, P_next, S_next])

        # Remove NaN-ed indices
        index_next, X_next, Y_next, P_next, S_next = remove_nan_rows([index_next, X_next, Y_next, P_next, S_next])

        self.index = np.append(self.index, index_next, axis=0) if self.index is not None else index_next

        self.Xs = np.append(self.Xs, X_next, axis=0) if self.Xs is not None else X_next
        self.Ys = np.append(self.Ys, Y_next, axis=0) if self.Ys is not None else Y_next
        if P_next is not None:
            self.Ps = np.append(self.Ps, P_next, axis=0) if self.Ps is not None else P_next
        if S_next is not None:
            self.Ss = np.append(self.Ss, S_next, axis=0) if self.Ss is not None else S_next

        # Lazily assign names on first update if they were not set at init
        if self.index_names is None:
            self.index_names = self.auto_naming_and_check(None, self.index.shape, default_prefix="index_")
        if self.X_names is None:
            self.X_names = self.auto_naming_and_check(None, self.Xs.shape, default_prefix="x_")
        if self.Y_names is None:
            self.Y_names = self.auto_naming_and_check(None, self.Ys.shape, default_prefix="y_")
        if self.P_names is None and self.Ps is not None:
            self.P_names = self.auto_naming_and_check(None, self.Ps.shape, default_prefix="p_")
        if self.S_names is None and self.Ss is not None:
            self.S_names = self.auto_naming_and_check(None, self.Ss.shape, default_prefix="s_")

        if refit_scaler and self.Ys.shape[0] > 1:
            self.Y_scaler.fit(self.Ys)

        if self.l_MultiFidelity:
            Ys_target = self.Ys[self.Ss[:, 0] == self.target_fidelity, :]
        else:
            Ys_target = self.Ys.copy()

        self.best_value = Ys_target.max(axis=0)
        if self.Ys.shape[0] > 1:
            self.best_value_transformed = self.Y_scaler.transform(self.best_value, return_torch=False)
        else:
            self.best_value_transformed = self.best_value.copy()

        self.nsamples = self.Ys.shape[0]

    def get_bounds(self):
        bounds = self.input_domain.get_bounds()
        if self.l_MultiFidelity:
            fidelity_bounds = self.fidelity_domain.get_bounds()
            bounds = np.concatenate([bounds, fidelity_bounds.reshape(-1, 1)], axis=1)
        return torch.tensor(bounds, dtype=dtype, device=device)

    def transform_XY(self):
        # Transform to [0,1]^d
        Xs_unit = self.input_domain.transform(self.Xs)
        # Append fidelities if multi-fidelity
        if self.l_MultiFidelity:
            Xs_unit = np.c_[Xs_unit, self.Ss]

        # Convert to torch tensors
        X_torch = torch.tensor(Xs_unit, dtype=dtype, device=device)
        # Y transformation, e.g. standardise
        Y_torch = self.Y_scaler.transform(self.Ys)

        return X_torch, Y_torch

    def transform_X(self, X):
        # Transform to input domain
        unit_X = self.input_domain.transform(X)
        if self.l_MultiFidelity:
            # X for each fidelity
            Ss = np.tile(np.array(self.fidelity_domain.fidelities), (unit_X.shape[0],)).reshape(-1, 1)
            unit_X = np.repeat(unit_X, repeats=self.fidelity_domain.num_fidelities, axis=0)
            unit_X = np.c_[unit_X, Ss]

        return unit_X

    def inverse_transform_X(self, unit_X):
        # Transform to input domain
        X = self.input_domain.inverse_transform(unit_X)

        return X

    def inverse_transform_Y(self, scaled_Ys, scaled_Y_stds=None):
        # Inverse transform Y values to original space
        unscaled_Ys, unscaled_stds = self.Y_scaler.inverse_transform(scaled_Ys, scaled_Y_stds, return_torch=False)
        return unscaled_Ys, unscaled_stds

    def fidelity_project(self, XSs):
        return self.fidelity_domain.project(XSs)

    def auto_naming_and_check(self, names, array_shape, default_prefix):
        if names is None:
            names = [f"{default_prefix}{i}" for i in range(array_shape[1])]
        else:  # Check shape
            assert len(names) == array_shape[1]
        return names

    def to_csv(self, filename: str):
        """
        Save index, Xs, Ps, Ss, Ys to a CSV file.
        Appends only new rows if file exists, comparing by index value to
        prevent overwriting rows with matching index values.
        """
        # Check arrays
        if self.index is None or self.Xs is None or self.Ys is None:
            raise ValueError("index, Xs, and Ys must not be None")

        # Combine data columns
        indices = check_for_2D_shape([self.index])[0]  # Ensure 2D index
        cols = [indices]
        header = []
        header += self.index_names

        if self.Ss is not None:
            cols.append(self.Ss)
            header += self.S_names
        if self.Xs is not None:
            cols.append(self.Xs)
            header += self.X_names
        if self.Ps is not None:
            cols.append(self.Ps)
            header += self.P_names
        if self.Ys is not None:
            cols.append(self.Ys)
            header += self.Y_names

        data = np.hstack(cols)
        n_index_cols = indices.shape[1]

        # Determine which rows have index values not already in the file
        if os.path.exists(filename):
            existing_indices = set()
            with open(filename, "r", newline="") as f:
                reader = csv.reader(f)
                next(reader, None)  # skip header
                for row in reader:
                    if row:
                        existing_indices.add(tuple(row[:n_index_cols]))

            # Convert index columns to string tuples matching csv.writer output
            new_idx_tuples = [tuple(str(v) for v in row) for row in data[:, :n_index_cols]]
            new_mask = np.array([t not in existing_indices for t in new_idx_tuples])
            new_data = data[new_mask]
        else:
            new_data = data

        if new_data.shape[0] == 0:
            print("No new rows to write.")
            return

        # Write or append
        mode = "a" if os.path.exists(filename) else "w"
        with open(filename, mode, newline="") as f:
            writer = csv.writer(f)
            if mode == "w":
                writer.writerow(header)
            writer.writerows(new_data)

    def save(self, filename: str):
        """
        Save state to a SQLite database file.

        Creates the database and table if they do not exist.
        Appends only new rows, using index columns as a unique key
        to skip duplicates (INSERT OR IGNORE).

        Column order: index, Ss (if present), Xs, Ps (if present), Ys.
        """
        if self.index is None or self.Xs is None or self.Ys is None:
            raise ValueError("index, Xs, and Ys must not be None")

        # ---- Build column definitions and data in canonical order ----
        columns = []  # list of (name, sqlite_type)
        arrays = []

        indices = check_for_2D_shape([self.index])[0]
        for name in self.index_names:
            columns.append((name, _numpy_dtype_to_sqlite(indices.dtype)))
        arrays.append(indices)

        if self.Ss is not None:
            for name in self.S_names:
                columns.append((name, _numpy_dtype_to_sqlite(self.Ss.dtype)))
            arrays.append(self.Ss)

        for name in self.X_names:
            columns.append((name, _numpy_dtype_to_sqlite(self.Xs.dtype)))
        arrays.append(self.Xs)

        if self.Ps is not None:
            for name in self.P_names:
                columns.append((name, _numpy_dtype_to_sqlite(self.Ps.dtype)))
            arrays.append(self.Ps)

        for name in self.Y_names:
            columns.append((name, _numpy_dtype_to_sqlite(self.Ys.dtype)))
        arrays.append(self.Ys)

        data = np.hstack(arrays)

        col_defs = ", ".join(f"{_quote_id(c[0])} {c[1]}" for c in columns)
        index_constraint = ", ".join(_quote_id(n) for n in self.index_names)
        quoted_col_names = ", ".join(_quote_id(c[0]) for c in columns)
        placeholders = ", ".join("?" for _ in columns)

        conn = sqlite3.connect(filename, timeout=30)
        try:
            conn.execute("PRAGMA journal_mode=WAL")

            # ---- Schema ----
            conn.execute(f"CREATE TABLE IF NOT EXISTS state ({col_defs}, UNIQUE({index_constraint}))")
            conn.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)")

            # ---- Metadata ----
            meta = {
                "input_domain": {
                    "type": type(self.input_domain).__name__,
                    "dim": int(self.input_domain.dim),
                    "b_low": self.input_domain.b_low.tolist(),
                    "b_up": self.input_domain.b_up.tolist(),
                    "steps": self.input_domain.steps.tolist(),
                },
                "column_names": {
                    "index_names": self.index_names,
                    "X_names": self.X_names,
                    "Y_names": self.Y_names,
                    "P_names": self.P_names,
                    "S_names": self.S_names,
                },
            }

            if self.fidelity_domain is not None:
                fd = self.fidelity_domain
                fd_meta = {
                    "num_fidelities": int(fd.num_fidelities),
                    "costs": getattr(fd, "costs", None),
                }
                if getattr(fd, "fidelities", None) is not None:
                    fd_meta["fidelities"] = list(fd.fidelities)
                if getattr(fd, "target_fidelity", None) is not None:
                    val = fd.target_fidelity
                    fd_meta["target_fidelity"] = val.tolist() if hasattr(val, "tolist") else val
                if getattr(fd, "fidelity_features", None) is not None:
                    val = fd.fidelity_features
                    fd_meta["fidelity_features"] = val.tolist() if hasattr(val, "tolist") else val
                meta["fidelity_domain"] = fd_meta

            for key, value in meta.items():
                conn.execute(
                    "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                    (key, json.dumps(value)),
                )

            # ---- Data rows (skip duplicates by index) ----
            insert_sql = f"INSERT OR IGNORE INTO state ({quoted_col_names}) VALUES ({placeholders})"
            conn.executemany(insert_sql, data.tolist())

            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def load(filename: str, Y_scaler: None | object = None, default_Ss: int | None = None):
        """
        Load state from a SQLite database file.

        Parameters
        ----------
        filename : str
            Path to the SQLite database.
        Y_scaler : object, optional
            Output scaler. A fresh StandardScaler is created when *None*.
        default_Ss : int, optional
            When not *None* and the loaded State has no ``Ss`` column,
            automatically set ``Ss`` to a constant array of this value
            and ``S_names`` to ``["fidelity"]``.

        Returns
        -------
        State
        """
        conn = sqlite3.connect(filename, timeout=30)
        try:
            conn.execute("PRAGMA journal_mode=WAL")

            # ---- Metadata ----
            meta = {}
            for key, value in conn.execute("SELECT key, value FROM metadata"):
                meta[key] = json.loads(value)

            # Reconstruct input domain
            id_meta = meta["input_domain"]
            domain_cls = ScaleFactorInputDomain if id_meta.get("type") == "ScaleFactorInputDomain" else InputDomain
            input_domain = domain_cls(
                dim=id_meta["dim"],
                b_low=np.array(id_meta["b_low"]),
                b_up=np.array(id_meta["b_up"]),
                steps=np.array(id_meta["steps"]),
            )

            # Reconstruct fidelity domain (optional)
            fidelity_domain = None
            if "fidelity_domain" in meta:
                fd_meta = meta["fidelity_domain"]
                fidelity_domain = FidelityDomain(
                    num_fidelities=fd_meta["num_fidelities"],
                    costs=fd_meta.get("costs"),
                )
                if fd_meta.get("fidelities") is not None:
                    fidelity_domain.fidelities = fd_meta["fidelities"]
                if fd_meta.get("target_fidelity") is not None:
                    fidelity_domain.target_fidelity = np.array(fd_meta["target_fidelity"])
                if fd_meta.get("fidelity_features") is not None:
                    fidelity_domain.fidelity_features = np.array(fd_meta["fidelity_features"])

            # Column names
            cn = meta["column_names"]
            index_names = cn["index_names"]
            X_names = cn["X_names"]
            Y_names = cn["Y_names"]
            P_names = cn.get("P_names")
            S_names = cn.get("S_names")

            # ---- Data ----
            cursor = conn.execute("SELECT * FROM state")
            rows = cursor.fetchall()

            if len(rows) == 0:
                return State(
                    input_domain=input_domain,
                    index=None,
                    Xs=None,
                    Ys=None,
                    index_names=index_names,
                    X_names=X_names,
                    Y_names=Y_names,
                    P_names=P_names,
                    S_names=S_names,
                    Y_scaler=Y_scaler,
                    fidelity_domain=fidelity_domain,
                )

            data = np.array(rows, dtype=float)
            db_col_names = [desc[0] for desc in cursor.description]

            def _extract(names):
                if names is None:
                    return None
                col_indices = [db_col_names.index(n) for n in names]
                return data[:, col_indices]

            state = State(
                input_domain=input_domain,
                index=_extract(index_names),
                Xs=_extract(X_names),
                Ys=_extract(Y_names),
                Ps=_extract(P_names),
                Ss=_extract(S_names),
                index_names=index_names,
                X_names=X_names,
                Y_names=Y_names,
                P_names=P_names,
                S_names=S_names,
                Y_scaler=Y_scaler,
                fidelity_domain=fidelity_domain,
            )

            if default_Ss is not None:
                if state.Ss is None and state.index is not None:
                    state.Ss = np.full((len(state.index), 1), default_Ss, dtype=float)
                    state.S_names = ["fidelity"]
                elif state.Ss is not None:
                    logging.getLogger("millefeuille.state").warning(
                        "default_Ss ignored: loaded State already has Ss"
                    )

            return state
        finally:
            conn.close()
