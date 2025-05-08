"""
Defines the optimiser state
"""

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

from .domain import InputDomain,FidelityDomain
import numpy as np
from dataclasses import dataclass

import numpy.typing as npt
from typing import Callable

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

        if(self.Ys is not None):
            self.best_value=self.Ys.max()
            self.best_value_transformed = self.Y_transform(self.best_value)
            self.nsamples = len(self.Ys)

    def update(self,index_next,X_next,Y_next,S_next=None,P_next=None):
        self.best_value = max(self.best_value, Y_next.max())
        self.best_value_transformed = self.Y_transform(self.best_value)

        self.index = np.append(self.index,index_next,axis=0)

        self.Xs = np.append(self.Xs,X_next,axis=0)
        self.Ys = np.append(self.Ys,Y_next,axis=0)
        if(P_next is not None):
            self.Xs = np.append(self.Xs,X_next,axis=0)
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