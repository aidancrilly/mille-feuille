import csv
import os
from dataclasses import dataclass

import h5py
import numpy as np
import numpy.typing as npt
import torch
from botorch.models.transforms.outcome import Standardize

from .definitions import device, dtype
from .domain import FidelityDomain, InputDomain

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

    # Find indices with NaNs across any of the valid arrays
    nan_mask = np.zeros(valid_arrays[0].shape[0], dtype=bool)
    for arr in valid_arrays:
        nan_mask |= np.isnan(arr).any(axis=1)

    # Remove rows at indices with NaNs from each non-None array
    cleaned_arrays = [arr[~nan_mask] if arr is not None else None for arr in arrays]
    return cleaned_arrays


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
    """Holds the complete state of an optimisation or sampling campaign.

    ``State`` is the central data container in ``mille-feuille``.  It stores
    all samples collected so far, tracks the best observed value, and provides
    utilities for transforming inputs/outputs for surrogate training.

    Parameters:
        input_domain: :class:`~millefeuille.domain.InputDomain` describing the
            continuous/discrete input space.
        index: Integer sample indices, shape ``(N, 1)`` or ``(N,)``.
        Xs: Input (parameter) samples, shape ``(N, dim)``.
        Ys: Objective function values, shape ``(N, n_objectives)``.
        Ps: Optional additional scalar outputs, shape ``(N, n_params)`` or
            ``None``.
        Ss: Optional fidelity labels for each sample, shape ``(N, 1)`` or
            ``None`` for single-fidelity problems.
        index_names: Column names for ``index``; auto-generated if ``None``.
        X_names: Column names for ``Xs``; auto-generated if ``None``.
        Y_names: Column names for ``Ys``; auto-generated if ``None``.
        P_names: Column names for ``Ps``; auto-generated if ``None``.
        S_names: Column names for ``Ss``; auto-generated if ``None``.
        Y_scaler: Output-space transformation (default:
            :class:`StandardScaler`).  Fitted automatically during
            :meth:`__post_init__`.
        fidelity_domain: :class:`~millefeuille.domain.FidelityDomain` for
            multi-fidelity problems; ``None`` for single-fidelity.
        nsamples: Number of samples stored (populated automatically).
        best_value: Best observed (raw) objective value seen so far.
        best_value_transformed: Best objective value in scaled space.

    Attributes:
        dim: Dimensionality of the input space (from ``input_domain``).
        l_MultiFidelity: ``True`` when a :attr:`fidelity_domain` is provided.
        target_fidelity: Index of the target fidelity (multi-fidelity only).
        fidelity_features: Per-fidelity feature dicts (multi-fidelity only).

    Example:
        >>> import numpy as np
        >>> from millefeuille.domain import InputDomain
        >>> from millefeuille.state import State
        >>> domain = InputDomain(
        ...     dim=1, b_low=np.array([0.0]), b_up=np.array([1.0]),
        ...     steps=np.array([0.0])
        ... )
        >>> state = State(
        ...     input_domain=domain,
        ...     index=np.array([0, 1, 2]),
        ...     Xs=np.array([[0.1], [0.5], [0.9]]),
        ...     Ys=np.array([[1.0], [3.0], [2.0]]),
        ... )
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
        self.index_names = self.auto_naming_and_check(self.index_names, self.index.shape, default_prefix="index_")
        self.X_names = self.auto_naming_and_check(self.X_names, self.Xs.shape, default_prefix="x_")
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

    def update(self, index_next, X_next, Y_next, S_next=None, P_next=None, refit_scaler=True):
        """Append new samples to the state and update tracking statistics.

        Parameters:
            index_next: Indices for the new samples, shape ``(B,)`` or
                ``(B, 1)``.
            X_next: New input samples, shape ``(B, dim)``.
            Y_next: New objective values, shape ``(B, n_objectives)``.
            S_next: Fidelity labels for the new samples, shape ``(B, 1)`` or
                ``None`` for single-fidelity problems.
            P_next: Additional scalar outputs, shape ``(B, n_params)`` or
                ``None``.
            refit_scaler: If ``True`` (default), re-fit :attr:`Y_scaler` on
                all ``Ys`` after appending.
        """
        # Check for 1D arrays
        index_next, X_next, Y_next, P_next, S_next = check_for_2D_shape([index_next, X_next, Y_next, P_next, S_next])

        # Remove NaN-ed indices
        index_next, X_next, Y_next, P_next, S_next = remove_nan_rows([index_next, X_next, Y_next, P_next, S_next])

        self.index = np.append(self.index, index_next, axis=0)

        self.Xs = np.append(self.Xs, X_next, axis=0)
        self.Ys = np.append(self.Ys, Y_next, axis=0)
        if P_next is not None:
            self.Ps = np.append(self.Ps, P_next, axis=0)
        if S_next is not None:
            self.Ss = np.append(self.Ss, S_next, axis=0)

        if refit_scaler and self.Ys.shape[0] > 1:
            self.Y_scaler.fit(self.Ys)

        if self.l_MultiFidelity:
            Ys_target = self.Ys[self.Ss[:, 0] == self.target_fidelity, :]
        else:
            Ys_target = self.Ys.copy()
        self.best_value = Ys_target.max(axis=0)
        self.best_value_transformed = self.Y_scaler.transform(self.best_value, return_torch=False)

        self.nsamples = self.Ys.shape[0]

    def get_bounds(self):
        """Return the optimisation bounds as a ``(2, d)`` PyTorch tensor.

        For single-fidelity problems ``d == dim``; for multi-fidelity problems
        the fidelity bounds are appended as an extra column so ``d == dim + 1``.

        Returns:
            torch.Tensor: Bounds tensor of shape ``(2, d)`` on the appropriate
            device and dtype, ready to pass to BoTorch optimisers.
        """
        bounds = self.input_domain.get_bounds()
        if self.l_MultiFidelity:
            fidelity_bounds = self.fidelity_domain.get_bounds()
            bounds = np.concatenate([bounds, fidelity_bounds.reshape(-1, 1)], axis=1)
        return torch.tensor(bounds, dtype=dtype, device=device)

    def transform_XY(self):
        """Return training tensors ``(X, Y)`` ready for surrogate fitting.

        Inputs are scaled to ``[0, 1]^dim`` via :attr:`input_domain`.  For
        multi-fidelity problems, fidelity labels are appended as an extra
        column.  Outputs are transformed via :attr:`Y_scaler` (standardised by
        default).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(X_torch, Y_torch)`` where
            ``X_torch`` has shape ``(N, dim)`` (or ``(N, dim + 1)`` for
            multi-fidelity) and ``Y_torch`` has shape ``(N, n_objectives)``.
        """
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
        """Scale raw input samples for use with the surrogate model.

        For multi-fidelity problems, the scaled inputs are tiled across all
        fidelity levels and the corresponding fidelity index is appended as a
        final column.

        Parameters:
            X: Raw input array of shape ``(N, dim)`` in parameter space.

        Returns:
            np.ndarray: Scaled array.  For single-fidelity: shape ``(N, dim)``;
            for multi-fidelity: shape ``(N * num_fidelities, dim + 1)``.
        """
        # Transform to input domain
        unit_X = self.input_domain.transform(X)
        if self.l_MultiFidelity:
            # X for each fidelity
            Ss = np.tile(np.array(self.fidelity_domain.fidelities), (unit_X.shape[0],)).reshape(-1, 1)
            unit_X = np.repeat(unit_X, repeats=self.fidelity_domain.num_fidelities, axis=0)
            unit_X = np.c_[unit_X, Ss]

        return unit_X

    def inverse_transform_X(self, unit_X):
        """Map unit-hypercube inputs back to parameter space.

        Parameters:
            unit_X: Scaled input array of shape ``(N, dim)`` in
                ``[0, 1]^dim``.

        Returns:
            np.ndarray: Array of shape ``(N, dim)`` in parameter space.
        """
        # Transform to input domain
        X = self.input_domain.inverse_transform(unit_X)

        return X

    def inverse_transform_Y(self, scaled_Ys, scaled_Y_stds=None):
        """Invert the output-space transformation.

        Reverses the :attr:`Y_scaler` to return predictions in the original
        (unscaled) objective space.

        Parameters:
            scaled_Ys: Scaled mean predictions, shape ``(N, n_objectives)``.
            scaled_Y_stds: Scaled standard-deviation predictions, shape
                ``(N, n_objectives)``.  Required when the scaler was fitted.

        Returns:
            tuple[np.ndarray, np.ndarray]: ``(unscaled_Ys, unscaled_stds)``
            both of shape ``(N, n_objectives)``.
        """
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
        """Save (or incrementally append) samples to a CSV file.

        On the first call a new file is created with a header row.  On
        subsequent calls only newly added rows (i.e. rows beyond those already
        written) are appended, so the method is safe to call after every
        :meth:`update` without duplicating data.

        The column order is: ``index`` → ``Ss`` (if present) → ``Xs`` →
        ``Ps`` (if present) → ``Ys``.

        Parameters:
            filename: Path to the output CSV file.

        Raises:
            ValueError: If :attr:`index`, :attr:`Xs`, or :attr:`Ys` is
                ``None``.
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

        # Determine how many rows already exist
        existing_rows = 0
        if os.path.exists(filename):
            with open(filename, "r", newline="") as f:
                existing_rows = sum(1 for _ in f) - 1  # subtract header row

        new_data = data[existing_rows:]

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
        """Serialise the state to an HDF5 file.

        Stores the input domain bounds, optional fidelity domain metadata, all
        sample arrays (``index``, ``Xs``, ``Ys``, ``Ps``, ``Ss``), column
        names, and tracking statistics (``nsamples``, ``best_value``,
        ``best_value_transformed``).

        Use :meth:`load` to restore the state from file.

        Parameters:
            filename: Path for the output ``.h5`` / ``.hdf5`` file.
        """
        # Use variable-length UTF-8 strings for names
        str_dt = h5py.string_dtype(encoding="utf-8")

        with h5py.File(filename, "w") as f:
            # --------------------
            # Input domain
            # --------------------
            grp_in = f.create_group("input_domain")
            grp_in.create_dataset("b_low", data=self.input_domain.b_low)
            grp_in.create_dataset("b_up", data=self.input_domain.b_up)
            grp_in.create_dataset("steps", data=self.input_domain.steps)
            f.attrs["input_dim"] = int(self.input_domain.dim)

            # --------------------
            # Fidelity domain (optional)
            # --------------------
            if self.fidelity_domain is not None:
                grp_fd = f.create_group("fidelity_domain")
                grp_fd.attrs["num_fidelities"] = int(self.fidelity_domain.num_fidelities)

                # Store what we can, if present on the object
                if getattr(self.fidelity_domain, "costs", None) is not None:
                    grp_fd.create_dataset("costs", data=np.asarray(self.fidelity_domain.costs))
                if getattr(self.fidelity_domain, "fidelities", None) is not None:
                    grp_fd.create_dataset("fidelities", data=np.asarray(self.fidelity_domain.fidelities))
                if getattr(self.fidelity_domain, "target_fidelity", None) is not None:
                    grp_fd.create_dataset("target_fidelity", data=np.asarray(self.fidelity_domain.target_fidelity))
                if getattr(self.fidelity_domain, "fidelity_features", None) is not None:
                    grp_fd.create_dataset("fidelity_features", data=np.asarray(self.fidelity_domain.fidelity_features))
                f.attrs["has_fidelity_domain"] = True
            else:
                f.attrs["has_fidelity_domain"] = False

            # --------------------
            # Arrays (only create datasets if present)
            # --------------------
            for name in ("index", "Xs", "Ys", "Ps", "Ss"):
                arr = getattr(self, name)
                if arr is not None:
                    f.create_dataset(name, data=arr)

            # --------------------
            # Column name lists
            # --------------------
            names_map = {
                "index_names": self.index_names,
                "X_names": self.X_names,
                "Y_names": self.Y_names,
                "P_names": self.P_names,
                "S_names": self.S_names,
            }
            for key, val in names_map.items():
                if val is not None:
                    # store as variable-length UTF-8 strings
                    f.create_dataset(key, data=np.array(val, dtype=object), dtype=str_dt)

            # --------------------
            # Metadata
            # --------------------
            f.attrs["nsamples"] = int(self.nsamples)

            # Save best values as datasets (vector-friendly).
            # Also keep attrs for backward compatibility/readability.
            if self.best_value is not None:
                f.create_dataset("best_value", data=np.asarray(self.best_value))
                try:
                    # If vector, HDF5 can still store as attr; if it fails, we skip.
                    f.attrs["best_value"] = np.asarray(self.best_value)
                except Exception:
                    pass

            if self.best_value_transformed is not None:
                f.create_dataset("best_value_transformed", data=np.asarray(self.best_value_transformed))
                try:
                    f.attrs["best_value_transformed"] = np.asarray(self.best_value_transformed)
                except Exception:
                    pass

    @staticmethod
    def load(filename: str, Y_scaler: None | object):
        """Restore a :class:`State` from an HDF5 file created by :meth:`save`.

        Parameters:
            filename: Path to the ``.h5`` / ``.hdf5`` file.
            Y_scaler: Output-space scaler to use after loading.  Pass ``None``
                to use the default :class:`StandardScaler`.

        Returns:
            State: A fully initialised :class:`State` instance.
        """
        def _read_list_of_str(fobj, key):
            if key in fobj:
                raw = fobj[key][:]
                # Handle bytes vs str
                return [s.decode("utf-8") if isinstance(s, bytes | np.bytes_) else str(s) for s in raw]
            return None

        with h5py.File(filename, "r") as f:
            # --------------------
            # Input domain
            # --------------------
            input_domain = InputDomain(
                dim=int(f.attrs["input_dim"]),
                b_low=f["input_domain/b_low"][:],
                b_up=f["input_domain/b_up"][:],
                steps=f["input_domain/steps"][:],
            )

            # --------------------
            # Fidelity domain (optional)
            # --------------------
            fidelity_domain = None
            if bool(f.attrs.get("has_fidelity_domain", False)):
                grp = f["fidelity_domain"]
                # Older files might only have costs; newer can have more fields
                costs = grp["costs"][:].tolist() if "costs" in grp else None
                fidelity_domain = FidelityDomain(
                    num_fidelities=int(grp.attrs["num_fidelities"]),
                    costs=costs,
                )
                # Fill optional fields when available
                if "fidelities" in grp:
                    fidelity_domain.fidelities = grp["fidelities"][:].tolist()
                if "target_fidelity" in grp:
                    fidelity_domain.target_fidelity = grp["target_fidelity"][:]
                if "fidelity_features" in grp:
                    fidelity_domain.fidelity_features = grp["fidelity_features"][:]

            # --------------------
            # Arrays
            # --------------------
            index = f["index"][:] if "index" in f else None
            Xs = f["Xs"][:] if "Xs" in f else None
            Ys = f["Ys"][:] if "Ys" in f else None
            Ps = f["Ps"][:] if "Ps" in f else None
            Ss = f["Ss"][:] if "Ss" in f else None

            # --------------------
            # Column name lists (optional)
            # --------------------
            index_names = _read_list_of_str(f, "index_names")
            X_names = _read_list_of_str(f, "X_names")
            Y_names = _read_list_of_str(f, "Y_names")
            P_names = _read_list_of_str(f, "P_names")
            S_names = _read_list_of_str(f, "S_names")

            # --------------------
            # Metadata
            # --------------------
            nsamples = int(f.attrs.get("nsamples", Ys.shape[0] if Ys is not None else 0))

            # Prefer datasets (vector-safe); fall back to (older) attrs when needed
            if "best_value" in f:
                best_value = f["best_value"][:]
            else:
                # Older files may have stored a scalar attr
                best_value = np.asarray(f.attrs.get("best_value", -float("inf")))

            if "best_value_transformed" in f:
                best_value_transformed = f["best_value_transformed"][:]
            else:
                best_value_transformed = np.asarray(f.attrs.get("best_value_transformed", -float("inf")))

            return State(
                input_domain=input_domain,
                index=index,
                Xs=Xs,
                Ys=Ys,
                Ps=Ps,
                Ss=Ss,
                index_names=index_names,
                X_names=X_names,
                Y_names=Y_names,
                P_names=P_names,
                S_names=S_names,
                Y_scaler=Y_scaler,
                fidelity_domain=fidelity_domain,
                nsamples=nsamples,
                best_value=best_value,
                best_value_transformed=best_value_transformed,
            )
