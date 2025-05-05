"""
Define the domain of the problem and scale to the unit hypercube (and back)
"""

import math
import numpy as np
from dataclasses import dataclass

@dataclass
class Domain:
    """
    Defines a problem domain which can have both discrete and continuous dimensions

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