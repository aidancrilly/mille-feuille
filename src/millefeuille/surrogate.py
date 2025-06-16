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
    def init(self, state: State):
        """
        Train the surrogate model on the current optimisation state.
        """
        pass

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
        Set internal models to evaluation mode (if using PyTorch).
        """
        self.model.eval()
        self.likelihood.eval()

    def save(self, filepath: str):
        """
        Save trained model to disk.
        """
        torch.save({'model_state_dict' : self.model.state_dict(),
            'likelihood_state_dict' : self.likelihood.state_dict()},
            filepath)

    def load(self, filepath: str, eval = True):
        """
        Load a saved model.
        """
        checkpoint = torch.load(filepath, weights_only=False, map_location = device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
        self.state_dicts = {'model_state_dict' : self.model.state_dict(),'likelihood_state_dict' : self.likelihood.state_dict()}
        if(eval):
            self.eval()

class SingleFidelityGPSurrogate(BaseGPSurrogate):

    def __init__(self):
        self.model = None
        self.likelihood = None
        self.state_dicts = None

    def get_XY(self,state: State):
        X_torch,Y_torch = state.transform_XY()
        assert X_torch.min() >= 0.0 and X_torch.max() <= 1.0 and torch.all(torch.isfinite(Y_torch))

        return X_torch,Y_torch

    def init(self,state: State,noise_interval=DEFAULT_NOISE_INTERVAL,lengthscale_interval=DEFAULT_LENGTHSCALE_INTERVAL, **kwargs):
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

        if(self.state_dicts is None):
            self.state_dicts = {'model_state_dict' : self.model.state_dict(),'likelihood_state_dict' : self.likelihood.state_dict()}
        else:
            self.model.load_state_dict(self.state_dicts['model_state_dict'])
            self.likelihood.load_state_dict(self.state_dicts['likelihood_state_dict'])

    def fit(self,state: State,approx_mll=False, **kwargs):
        self.init(state)

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
        return {'mean' : mean, 'std' : std}


class MultiFidelityGPSurrogate(BaseGPSurrogate):

    def __init__(self):
        self.model = None
        self.likelihood = None
        self.state_dicts = None

    def get_XY(self,state):
        X_torch,Y_torch = state.transform_XY()
        assert X_torch[:,:-1].min() >= 0.0 and X_torch[:,:-1].max() <= 1.0 and torch.all(torch.isfinite(Y_torch))

        return X_torch,Y_torch

    def init(self,state,noise_interval=DEFAULT_NOISE_INTERVAL,lengthscale_interval=DEFAULT_LENGTHSCALE_INTERVAL):
        X_torch,Y_torch = self.get_XY(state)

        self.likelihood = GaussianLikelihood(noise_constraint=Interval(*noise_interval))

        self.model = ModifiedSingleTaskMultiFidelityGP(
                X_torch, Y_torch,
                likelihood=self.likelihood, outcome_transform= None
            )
        
        if(self.state_dicts is None):
            self.state_dicts = {'model_state_dict' : self.model.state_dict(),'likelihood_state_dict' : self.likelihood.state_dict()}
        else:
            self.model.load_state_dict(self.state_dicts['model_state_dict'])
            self.likelihood.load_state_dict(self.state_dicts['likelihood_state_dict'])

    def fit(self,state: State,approx_mll=False, **kwargs):
        self.init(state, **kwargs)

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)

        # Fit the model
        fit_gpytorch_mll(mll, approx_mll=approx_mll)

    def predict(self, state: State, Xs):
        Xs_unit = state.transform_X(Xs)
        test_X = torch.tensor(Xs_unit, dtype=torch.double, device=device)
        with torch.no_grad():
            post = self.model.likelihood(self.model(test_X))
            mean = post.mean.cpu().numpy().reshape(-1, state.fidelity_domain.num_fidelities)
            var = post.variance.cpu().numpy().reshape(-1, state.fidelity_domain.num_fidelities)
            std = np.sqrt(var)
        mean, std = state.inverse_transform_Y(mean, std)
        return {fid : {'mean' : mean[:,fid], 'std' : std[:,fid]} for fid in range(state.fidelity_domain.num_fidelities)}
    
