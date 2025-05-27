"""
Defines the optimiser state
"""

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

from .domain import InputDomain,FidelityDomain

import os
import csv
import h5py
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from typing import Callable

def check_for_2D_shape(arrays):
    # Ensure correct shape from each non-None array
    cleaned_arrays = []
    for arr in arrays:
        if arr is not None:
            if arr.ndim == 1:
                cleaned_arrays.append(arr.reshape(-1,1))
            elif arr.ndim == 2:
                cleaned_arrays.append(arr)
            else:
                print('WARNING: State recieved non-1 or -2D array!')
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
    cleaned_arrays = [
        arr[~nan_mask] if arr is not None else None
        for arr in arrays
    ]
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

    Y_transform = transformation on output space for training, e.g. standardise

    """
    input_domain: type[InputDomain]
    index : None | npt.NDArray
    Xs : None | npt.NDArray
    Ys : None | npt.NDArray
    Ps : None | npt.NDArray = None
    Ss : None | npt.NDArray = None

    Y_transform : None | Callable = None

    fidelity_domain: None | type[FidelityDomain] = None

    nsamples : int = 0
    best_value: float = -float("inf")
    best_value_transformed: float = -float("inf")
        
    def __post_init__(self):
        self.dim = self.input_domain.dim

        if(self.Y_transform is None):
            self.Y_transform = lambda x : x

        if(self.fidelity_domain is not None):
            self.fidelity_domain.combine_with_input_domain(self.dim)
            self.target_fidelity = self.fidelity_domain.target_fidelity
            self.fidelity_features = self.fidelity_domain.fidelity_features
            self.l_MultiFidelity = True
        else:
            self.l_MultiFidelity = False

        # Check for 1D arrays
        self.index, self.Xs, self.Ys, self.Ps, self.Ss = check_for_2D_shape([self.index, self.Xs, self.Ys, self.Ps, self.Ss])

        # Remove NaN-ed indices
        self.index, self.Xs, self.Ys, self.Ps, self.Ss = remove_nan_rows([self.index, self.Xs, self.Ys, self.Ps, self.Ss])

        if(self.Ys is not None):
            self.best_value=self.Ys.max()
            self.best_value_transformed = self.Y_transform(self.best_value)
            self.nsamples = len(self.Ys)

    def update(self,index_next,X_next,Y_next,S_next=None,P_next=None):
        # Check for 1D arrays
        index_next, X_next, Y_next, P_next, S_next = check_for_2D_shape([index_next, X_next, Y_next, P_next, S_next])

        # Remove NaN-ed indices
        index_next, X_next, Y_next, P_next, S_next = remove_nan_rows([index_next, X_next, Y_next, P_next, S_next])

        self.best_value = max(self.best_value, Y_next.max())
        self.best_value_transformed = self.Y_transform(self.best_value)

        self.index = np.append(self.index,index_next,axis=0)

        self.Xs = np.append(self.Xs,X_next,axis=0)
        self.Ys = np.append(self.Ys,Y_next,axis=0)
        if(P_next is not None):
            self.Ps = np.append(self.Ps,P_next,axis=0)
        if(S_next is not None):
            self.Ss = np.append(self.Ss,S_next,axis=0)

        self.nsamples = len(self.Ys)

    def get_bounds(self):
        bounds = self.input_domain.get_bounds()
        if(self.l_MultiFidelity):
            fidelity_bounds = self.fidelity_domain.get_bounds()
            bounds = np.concatenate([bounds,fidelity_bounds.reshape(-1,1)],axis=1)
        return torch.tensor(bounds, dtype=dtype, device=device)

    def transform_XY(self):
        # Transform to [0,1]^d
        Xs_unit = self.input_domain.transform(self.Xs)
        # Append fidelities if multi-fidelity
        if(self.l_MultiFidelity):
            Xs_unit = np.c_[Xs_unit,self.Ss]

        # Y transformation, e.g. standardise
        train_Y = self.Y_transform(self.Ys)

        # Convert to torch tensors
        X_torch = torch.tensor(Xs_unit, dtype=dtype, device=device)
        Y_torch = torch.tensor(train_Y, dtype=dtype, device=device)

        return X_torch,Y_torch

    def inverse_transform_X(self,unit_X):
        # Transform to input domain
        X = self.input_domain.inverse_transform(unit_X)

        return X

    def fidelity_project(self,XSs):
        return self.fidelity_domain.project(XSs)

    def save(self, filename: str):
        with h5py.File(filename, 'w') as f:
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
    def load(filename: str, Y_transform : None | Callable):
        with h5py.File(filename, 'r') as f:
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
                    num_fidelities=int(grp.attrs["num_fidelities"]),
                    costs=grp["costs"][:].tolist()
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
                Y_transform=Y_transform,
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
        cols = [self.index.reshape(-1, 1)]  # Ensure 2D index
        header = ["index"]

        if self.Xs is not None:
            cols.append(self.Xs)
            header += [f"x{i}" for i in range(self.Xs.shape[1])]
        if self.Ps is not None:
            cols.append(self.Ps)
            header += [f"p{i}" for i in range(self.Ps.shape[1])]
        if self.Ss is not None:
            cols.append(self.Ss)
            header += [f"s{i}" for i in range(self.Ss.shape[1])]
        if self.Ys is not None:
            cols.append(self.Ys)
            header += [f"y{i}" for i in range(self.Ys.shape[1])]

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
