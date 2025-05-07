"""
Define the domains of the problem
"""

import math
import numpy as np
from botorch.acquisition.utils import project_to_target_fidelity
from dataclasses import dataclass

@dataclass
class InputDomain:
    """
    Defines an input domain which can have both discrete and continuous dimensions

    This is scaled to the unit hypercube
    """
    dim: int
    b_low : np.ndarray
    b_up : np.ndarray
    steps : np.ndarray

    def __post_init__(self):
        self.discrete_indices = [i for i in range(self.dim) if self.steps[i] != 0.0]
        self.discrete_dim = len(self.discrete_indices)
        self.discrete_bound = [int((self.b_up[i]-self.b_low[i])/self.steps[i]) for i in self.discrete_indices]

    def get_bounds(self):
        lb = np.zeros(self.dim)
        ub = np.ones(self.dim)
        bounds = np.stack([lb, ub])
        return bounds

    def transform(self,X):
        X_scaled = X.copy()
        # Transform to [0,1]^d
        for n in range(self.dim):
            X_scaled[:,n] = (X_scaled[:,n]-self.b_low[n])/(self.b_up[n]-self.b_low[n])
        # Catch floating point errors
        X_scaled[np.isclose(X_scaled,0.0)] = 0.0
        X_scaled[np.isclose(X_scaled,1.0)] = 1.0
        return X_scaled

    def inverse_transform(self,X):
        X_scaled = X.copy()
        # Scale back to parameter space
        for n in range(self.dim):
            X_scaled[:,n] = (self.b_up[n]-self.b_low[n])*X_scaled[:,n]+self.b_low[n]
            # For discrete spaces, find nearest point
            if(n in self.discrete_indices):
                X_scaled[:,n] = np.rint(X_scaled[:,n]/self.steps[n])*self.steps[n]
        return X_scaled

@dataclass
class FidelityDomain:
    """
    Defines the fidelity space domain which is discrete

    One can assign costs to each fidelity here
    """

    num_fidelities: int
    costs : None | list

    def __post_init__(self):
        if(self.costs is None):
            self.costs = [i for i in range(self.num_fidelities)]
        else:
            assert len(self.costs) == self.num_fidelities

        self.minimal_fidelity  = 0
        self.target_fidelity   = self.num_fidelities
        self.fidelities        = [i for i in range(self.target_fidelity)]

    def combine_with_input_domain(self,input_dim):
        self.fidelity_weights  = {input_dim: 1.0}
        self.target_fidelities = {input_dim: self.target_fidelity}
        self.fidelity_features = [{input_dim: f} for f in self.fidelities]

    def get_bounds(self):
        bounds = np.array([self.minimal_fidelity, self.target_fidelity])
        return bounds

    def project(self,XSs):
        return project_to_target_fidelity(X=XSs, target_fidelities=self.target_fidelities)