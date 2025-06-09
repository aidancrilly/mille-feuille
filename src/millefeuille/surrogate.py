"""
Defines a number of surrogate models
"""

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

import numpy as np

from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from .kernel import ModifiedSingleTaskMultiFidelityGP

from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

# Default GP hyperparameters
DEFAULT_NOISE_INTERVAL = [1e-8,1e-5]
DEFAULT_LENGTHSCALE_INTERVAL = [0.005,4.0]

from .state import State
from abc import ABC, abstractmethod
import numpy.typing as npt

class BaseGPSurrogate(ABC):
    """
    Abstract base class for all surrogate models in mille-feuille.
    """

    @abstractmethod
    def fit(self, state: State):
        """
        Train the surrogate model on the current optimisation state.
        """
        pass

    @abstractmethod
    def predict(self, state: State, Xs: npt.NDArray):
        """
        Predict outputs for new input points.

        Returns:
            - If single objective: np.ndarray of shape (N, 2) -> mean, std
            - If multi-objective: dict {key: (mean, std)}
        """
        pass

    @abstractmethod
    def get_XY(self, state: State, output_key: str = None):
        """
        Extract training data from the state. Can handle single or multi-output.

        Returns:
            X_torch, Y_torch
        """
        pass

    def eval(self):
        """
        Optionally set internal models to evaluation mode (if using PyTorch).
        """
        pass

    def save(self, filepath: str):
        """
        Optionally save trained model to disk.
        """
        raise NotImplementedError("Saving not implemented for this surrogate.")

    def load(self, filepath: str):
        """
        Optionally load a saved model.
        """
        raise NotImplementedError("Loading not implemented for this surrogate.")

class SingleFidelityGPSurrogate(BaseGPSurrogate):

    def __init__(self):
        self.model = None
        self.likelihood = None

    def get_XY(self,state: State):
        X_torch,Y_torch = state.transform_XY()
        assert X_torch.min() >= 0.0 and X_torch.max() <= 1.0 and torch.all(torch.isfinite(Y_torch))

        return X_torch,Y_torch

    def fit(self,state: State,noise_interval=DEFAULT_NOISE_INTERVAL,lengthscale_interval=DEFAULT_LENGTHSCALE_INTERVAL,approx_mll=False):
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
        fit_gpytorch_mll(mll, approx_mll=approx_mll)

    def predict(self, state: State, Xs):
        Xs_unit = state.transform_X(Xs)
        test_X = torch.tensor(Xs_unit, dtype=torch.double, device=device)
        with torch.no_grad():
            post = self.likelihood(self.model(test_X))
            mean = post.mean.cpu().numpy().reshape(-1, 1)
            var = post.variance.cpu().numpy().reshape(-1, 1)
            std = np.sqrt(var)
        mean, std = state.inverse_transform_Y(mean, std)
        return np.concatenate([mean, std], axis=1)

    def eval(self):
        self.model.eval()
        self.likelihood.eval()

class MultiObjectiveSingleFidelityGPSurrogate(BaseGPSurrogate):
    """
    A surrogate model for multiple objectives using independent SingleTaskGPs.
    """

    def __init__(self, num_objective):
        self.num_objective = num_objective
        self.models = {}
        self.likelihoods = {}

    def get_XY(self, state: State):
        X_torch, Y_torch = state.transform_XY()
        return X_torch, Y_torch

    def fit(self, state: State, noise_interval=DEFAULT_NOISE_INTERVAL, lengthscale_interval=DEFAULT_LENGTHSCALE_INTERVAL,approx_mll=False):
        """
        Trains a GP for each output key separately.
        """
        X_torch, Y_torch = self.get_XY(state)
        for ikey in range(self.num_objective):
            
            likelihood = GaussianLikelihood(noise_constraint=Interval(*noise_interval))
            covar_module = ScaleKernel(
                MaternKernel(
                    nu=2.5, ard_num_dims=state.dim, lengthscale_constraint=Interval(*lengthscale_interval)
                )
            )
            model = SingleTaskGP(X_torch, Y_torch[:,ikey], covar_module=covar_module, likelihood=likelihood)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll, approx_mll=approx_mll)

            model.eval()
            likelihood.eval()
            self.models[ikey] = model
            self.likelihoods[ikey] = likelihood

    def predict(self, state: State, Xs):
        Xs_unit = state.transform_X(Xs)
        test_X = torch.tensor(Xs_unit, dtype=dtype, device=device)

        means = []
        stds = []
        for ikey in range(self.num_objective):
            model = self.models[ikey]
            likelihood = self.likelihoods[ikey]
            with torch.no_grad():
                post = likelihood(model(test_X))
                mean = post.mean.cpu().numpy().reshape(-1, 1)
                var = post.variance.cpu().numpy().reshape(-1, 1)
                std = np.sqrt(var)
                means.append(mean)
                stds.append(std)

        means, stds = state.inverse_transform_Y(np.array(means),np.array(stds))

        predictions = {}
        for ikey in range(self.num_objective):
            mean, std = means[:,ikey], stds[:,ikey]
            predictions[ikey] = (mean, std)

        return predictions
class MultiFidelityGPSurrogate:

    def __init__(self):
        self.model = None
        self.likelihood = None

    def get_XY(self,state):
        X_torch,Y_torch = state.transform_XY()
        assert X_torch[:,:-1].min() >= 0.0 and X_torch[:,:-1].max() <= 1.0 and torch.all(torch.isfinite(Y_torch))

        return X_torch,Y_torch

    def fit(self,state,noise_interval=DEFAULT_NOISE_INTERVAL,lengthscale_interval=DEFAULT_LENGTHSCALE_INTERVAL,approx_mll=False):
        X_torch,Y_torch = self.get_XY(state)

        self.likelihood = GaussianLikelihood(noise_constraint=Interval(*noise_interval))

        self.model = ModifiedSingleTaskMultiFidelityGP(
                X_torch, Y_torch,
                likelihood=self.likelihood, outcome_transform= None
            )
        
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)

        # Fit the model
        fit_gpytorch_mll(mll, approx_mll=approx_mll)

    def predict(self, state: State, Xs):
        Xs_unit = state.transform_X(Xs)
        test_X = torch.tensor(Xs_unit, dtype=torch.double, device=device)
        with torch.no_grad():
            post = self.likelihood(self.model(test_X))
            mean = post.mean.cpu().numpy().reshape(-1, state.fidelity_domain.num_fidelities)
            var = post.variance.cpu().numpy().reshape(-1, state.fidelity_domain.num_fidelities)
            std = np.sqrt(var)
        mean, std = state.inverse_transform_Y(mean, std)
        return {'mean' : mean, 'std' : std}

    def eval(self):
        self.model.eval()
        self.likelihood.eval()