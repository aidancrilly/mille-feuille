"""
Defines a number of surrogate models
"""
import torch

from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP

from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

# Default GP hyperparameters
DEFAULT_NOISE_INTERVAL = [1e-8,1e-5]
DEFAULT_LENGTHSCALE_INTERVAL = [0.005,4.0]

class SingleFidelityGPSurrogate:

    def __init__(self):
        self.model = None
        self.likelihood = None

    def get_XY(self,state):
        X_torch,Y_torch = state.transform_XY()
        assert X_torch.min() >= 0.0 and X_torch.max() <= 1.0 and torch.all(torch.isfinite(Y_torch))

        return X_torch,Y_torch

    def fit(self,state,noise_interval=DEFAULT_NOISE_INTERVAL,lengthscale_interval=DEFAULT_LENGTHSCALE_INTERVAL):
        X_torch,Y_torch = self.get_XY(state)

        self.likelihood = GaussianLikelihood(noise_constraint=Interval(*noise_interval))
        covar_module = ScaleKernel( 
            MaternKernel(
                nu=2.5, ard_num_dims=state.dim, lengthscale_constraint=Interval(*lengthscale_interval)
            )
        )
        self.model = SingleTaskGP(
                X_torch, Y_torch, covar_module=covar_module, likelihood=self.likelihood
            )
        
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)

        # Fit the model
        fit_gpytorch_mll(mll)

class MultiFidelityGPSurrogate:

    def __init__(self):
        self.model = None
        self.likelihood = None

    def get_XY(self,state):
        X_torch,Y_torch = state.transform_XY()
        assert X_torch[:,:-1].min() >= 0.0 and X_torch[:,:-1].max() <= 1.0 and torch.all(torch.isfinite(Y_torch))

        return X_torch,Y_torch

    def fit(self,state,noise_interval=DEFAULT_NOISE_INTERVAL,lengthscale_interval=DEFAULT_LENGTHSCALE_INTERVAL):
        X_torch,Y_torch = self.get_XY(state)

        self.likelihood = GaussianLikelihood(noise_constraint=Interval(*noise_interval))

        self.model = SingleTaskMultiFidelityGP(
                X_torch, Y_torch, data_fidelities = [state.dim], likelihood=self.likelihood
            )
        
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)

        # Fit the model
        fit_gpytorch_mll(mll)
