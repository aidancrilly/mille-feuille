import csv
import os
from dataclasses import dataclass

import h5py
import numpy as np
import numpy.typing as npt
import torch
from sklearn.preprocessing import StandardScaler

from .domain import FidelityDomain, InputDomain

"""
Defines the optimiser state
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
dtype = torch.double


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
        self.index_names = self.auto_naming_and_check(self.index_names, self.index.shape, default_prefix="index_")
        self.X_names = self.auto_naming_and_check(self.X_names, self.Xs.shape, default_prefix="x_")
        self.Y_names = self.auto_naming_and_check(self.Y_names, self.Ys.shape, default_prefix="y_")
        if self.Ps is not None:
            self.P_names = self.auto_naming_and_check(self.P_names, self.Ps.shape, default_prefix="p_")
        if self.Ss is not None:
            self.S_names = self.auto_naming_and_check(self.S_names, self.Ss.shape, default_prefix="s_")

        if self.Ys is not None:
            self.Y_scaler.fit(self.Ys)
            self.best_value = self.Ys.max(axis=0)
            self.best_value_transformed = self.Y_scaler.transform([self.best_value])
            self.nsamples = self.Ys.shape[0]

    def update(self, index_next, X_next, Y_next, S_next=None, P_next=None, refit_scaler=True):
        # Check for 1D arrays
        index_next, X_next, Y_next, P_next, S_next = check_for_2D_shape([index_next, X_next, Y_next, P_next, S_next])

        # Remove NaN-ed indices
        index_next, X_next, Y_next, P_next, S_next = remove_nan_rows([index_next, X_next, Y_next, P_next, S_next])

        self.best_value = max(self.best_value, Y_next.max(axis=0))
        self.best_value_transformed = self.Y_scaler.transform([self.best_value])

        self.index = np.append(self.index, index_next, axis=0)

        self.Xs = np.append(self.Xs, X_next, axis=0)
        self.Ys = np.append(self.Ys, Y_next, axis=0)
        if P_next is not None:
            self.Ps = np.append(self.Ps, P_next, axis=0)
        if S_next is not None:
            self.Ss = np.append(self.Ss, S_next, axis=0)

        if refit_scaler and self.Ys.shape[0] > 1:
            self.Y_scaler.fit(self.Ys)

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

        # Y transformation, e.g. standardise
        train_Y = self.Y_scaler.transform(self.Ys)

        # Convert to torch tensors
        X_torch = torch.tensor(Xs_unit, dtype=dtype, device=device)
        Y_torch = torch.tensor(train_Y, dtype=dtype, device=device)

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
        """
        Inverse-transforms scaled Y values and optionally their std deviations.

        Parameters:
            scaled_Ys: np.ndarray, transformed Y values
            scaled_Y_stds: np.ndarray, std devs in transformed space (no mean shift)

        Returns:
            Tuple of (unscaled_Ys, unscaled_stds) or unscaled_Ys if stds not provided
        """
        unscaled_Ys = self.Y_scaler.inverse_transform(scaled_Ys)

        if scaled_Y_stds is not None:
            std_scale = np.sqrt(self.Y_scaler.var_) if hasattr(self.Y_scaler, "var_") else self.Y_scaler.scale_
            unscaled_stds = scaled_Y_stds * std_scale
            return unscaled_Ys, unscaled_stds

        return unscaled_Ys

    def fidelity_project(self, XSs):
        return self.fidelity_domain.project(XSs)

    def auto_naming_and_check(self, names, array_shape, default_prefix):
        if names is None:
            names = [f"{default_prefix}{i}" for i in range(array_shape[1])]
        else:  # Check shape
            assert len(names) == array_shape[1]
        return names

    def save(self, filename: str):
        with h5py.File(filename, "w") as f:
            # Input domain
            f.create_dataset("input_domain/b_low", data=self.input_domain.b_low)
            f.create_dataset("input_domain/b_up", data=self.input_domain.b_up)
            f.create_dataset("input_domain/steps", data=self.input_domain.steps)
            f.attrs["input_dim"] = self.input_domain.dim

            # Fidelity domain (optional)
            if self.fidelity_domain is not None:
                grp = f.create_group("fidelity_domain")
                grp.attrs["num_fidelities"] = self.fidelity_domain.num_fidelities
                grp.create_dataset("costs", data=self.fidelity_domain.costs)
                f.attrs["has_fidelity_domain"] = True
            else:
                f.attrs["has_fidelity_domain"] = False

            # Arrays
            f.create_dataset("index", data=self.index)
            f.create_dataset("Xs", data=self.Xs)
            f.create_dataset("Ys", data=self.Ys)
            if self.Ps is not None:
                f.create_dataset("Ps", data=self.Ps)
            if self.Ss is not None:
                f.create_dataset("Ss", data=self.Ss)

            # Scalar metadata
            f.attrs["nsamples"] = self.nsamples
            f.attrs["best_value"] = self.best_value
            f.attrs["best_value_transformed"] = self.best_value_transformed

    @staticmethod
    def load(filename: str, Y_scaler: None | object):
        with h5py.File(filename, "r") as f:
            # Input domain
            input_domain = InputDomain(
                dim=int(f.attrs["input_dim"]),
                b_low=f["input_domain/b_low"][:],
                b_up=f["input_domain/b_up"][:],
                steps=f["input_domain/steps"][:],
            )

            # Fidelity domain
            fidelity_domain = None
            if f.attrs["has_fidelity_domain"]:
                grp = f["fidelity_domain"]
                fidelity_domain = FidelityDomain(
                    num_fidelities=int(grp.attrs["num_fidelities"]), costs=grp["costs"][:].tolist()
                )

            # Arrays
            index = f["index"][:]
            Xs = f["Xs"][:]
            Ys = f["Ys"][:]
            Ps = f["Ps"][:] if "Ps" in f else None
            Ss = f["Ss"][:] if "Ss" in f else None

            return State(
                input_domain=input_domain,
                index=index,
                Xs=Xs,
                Ys=Ys,
                Ps=Ps,
                Ss=Ss,
                Y_scaler=Y_scaler,
                fidelity_domain=fidelity_domain,
                nsamples=int(f.attrs["nsamples"]),
                best_value=float(f.attrs["best_value"]),
                best_value_transformed=float(f.attrs["best_value_transformed"]),
            )

    def to_csv(self, filename: str):
        """
        Save index, Xs, Ps, Ss, Ys to a CSV file.
        Appends only new rows if file exists.
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
