"""
Defines the optimiser state
"""

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

from .domain import Domain
from botorch.acquisition.utils import project_to_target_fidelity
import numpy as np
from dataclasses import dataclass

import numpy.typing as npt
from typing import Callable

@dataclass
class State:
    """
    State containing information on problem and its progress

    domain = input space Domain

    index = each sample is given an index (doesn't necessarily just count the samples)
    Xs = input space samples
    Ys = cost function values
    Ps = additional scalar values to store
    Ss = model fidelities of samples

    Y_transform = transformation on output space for training, e.g. standardise

    """
    domain: type[Domain]
    
    index : None | npt.NDArray
    Xs : None | npt.NDArray
    Ys : None | npt.NDArray
    Ps : None | npt.NDArray
    Ss : None | npt.NDArray

    Y_transform = Callable

    costs : None | list

    nsamples : int = 0
    best_value: float = -float("inf")
    best_value_transformed: float = -float("inf")

    def __init__(self,
    domain,
    index,Xs,Ys,Ss=None,Ps=None,
    Y_transform=lambda x : x,
    costs=None):
        self.domain = domain
        self.dim = self.domain.dim

        self.index = index
        self.Xs = Xs
        self.Ys = Ys
        self.Ss = Ss
        self.Ps = Ps
        
        self.Y_transform = Y_transform
        
        if(costs is not None):
            self.l_MultiFidelity = True
            self.costs = costs
        else:
            self.l_MultiFidelity = False

        if(self.l_MultiFidelity):
            self.minimal_fidelity  = 0
            self.target_fidelity   = len(self.costs)
            self.fidelities        = [i for i in range(self.target_fidelity)]
            self.fidelity_weights  = {self.dim: 1.0}
            self.target_fidelities = {self.dim: self.target_fidelity}
            self.fidelity_features = [{self.dim: f} for f in self.fidelities]

        if(self.Ys is not None):
            self.best_value=self.Ys.max()
            self.best_value_transformed = self.Y_transform(self.best_value)
            self.nsamples = len(self.Ys)

    def update(self,X_next,Y_next,S_next=None,P_next=None):
        self.best_value = max(self.best_value, Y_next.max())
        self.best_value_transformed = self.Y_transform(self.best_value)

        self.Xs = np.append(self.Xs,X_next,axis=0)
        self.Ys = np.append(self.Ys,Y_next,axis=0)
        if(P_next is not None):
            self.Xs = np.append(self.Xs,X_next,axis=0)
        if(S_next is not None):
            self.Ss = np.append(self.Ss,S_next,axis=0)

        self.nsamples = len(self.Ys)

    def transform_XY(self):
        # Transform to [0,1]^d
        Xs_unit = self.domain.transform(self.Xs)
        # Append fidelities if multi-fidelity
        if(self.l_MultiFidelity):
            Xs_unit = np.c_[Xs_unit,self.Ss]

        # Y transformation, e.g. standardise
        train_Y = self.Y_transform(self.Ys)

        # Convert to torch tensors
        X_torch = torch.tensor(Xs_unit, dtype=dtype, device=device)
        Y_torch = torch.tensor(train_Y, dtype=dtype, device=device)

        return X_torch,Y_torch

    def get_bounds(self):
        lb = torch.zeros(self.dim)
        ub = torch.ones(self.dim)
        bounds = torch.stack([lb, ub])
        if(self.l_MultiFidelity):
            bounds = torch.concat([bounds,torch.Tensor([self.minimal_fidelity,self.target_fidelity]).unsqueeze(-1)],dim=1)
        return bounds

    def fidelity_project(self,XSs):
        return project_to_target_fidelity(X=XSs, target_fidelities=self.target_fidelities)