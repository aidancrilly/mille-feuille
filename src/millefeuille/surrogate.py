import copy
from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.ensemble import EnsembleModel
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.module import Module
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset, random_split

from .kernel import ModifiedSingleTaskMultiFidelityGP
from .state import State

"""
Defines a number of surrogate models
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double


class BaseSurrogate(ABC):
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

    def get_XY(self, state: State, output_key: str | None = None):
        """
        Extract training data from the state. Can handle single or multi-output.

        Returns:
            X_torch, Y_torch
        """
        X_torch, Y_torch = state.transform_XY()
        assert X_torch.min() >= 0.0 and X_torch.max() <= 1.0 and torch.all(torch.isfinite(Y_torch))

        return X_torch, Y_torch

    @abstractmethod
    def eval(self):
        """
        Set internal models to evaluation mode (PyTorch).
        """
        pass

    @abstractmethod
    def save(self, filepath: str):
        """
        Save trained model to disk.
        """
        pass

    @abstractmethod
    def load(self, filepath: str, eval=True):
        """
        Load a saved model.
        """
        pass


##################################################
############### Gaussian Processes ###############
##################################################

# Default GP hyperparameters
DEFAULT_NOISE_INTERVAL = [1e-8, 1e-5]
DEFAULT_LENGTHSCALE_INTERVAL = [0.005, 4.0]


class BaseGPSurrogate(BaseSurrogate, ABC):
    """
    Abstract base class for all GP surrogate models in mille-feuille.
    """

    def __init__(self):
        self.model = None
        self.mean_module = None
        self.likelihood = None
        self.state_dicts = None

    @abstractmethod
    def init_GP_model(self, state: State):
        """
        Train the surrogate model on the current optimisation state.
        """
        pass

    def update_state_dicts(self):
        """
        Update internal state dictionaries with current model parameters.
        """
        if self.state_dicts is None:
            self.state_dicts = {
                "model_state_dict": self.model.state_dict(),
                "likelihood_state_dict": self.likelihood.state_dict(),
            }
        else:
            self.model.load_state_dict(self.state_dicts["model_state_dict"])
            self.likelihood.load_state_dict(self.state_dicts["likelihood_state_dict"])

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
        torch.save(
            {"model_state_dict": self.model.state_dict(), "likelihood_state_dict": self.likelihood.state_dict()},
            filepath,
        )

    def load(self, filepath: str, eval=True):
        """
        Load a saved model.
        """
        checkpoint = torch.load(filepath, weights_only=False, map_location=device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.likelihood.load_state_dict(checkpoint["likelihood_state_dict"])
        self.state_dicts = {
            "model_state_dict": self.model.state_dict(),
            "likelihood_state_dict": self.likelihood.state_dict(),
        }
        if eval:
            self.eval()


class SingleFidelityGPSurrogate(BaseGPSurrogate):
    def init_GP_model(
        self,
        state: State,
        mean_module: Module | None = None,
        noise_interval=DEFAULT_NOISE_INTERVAL,
        lengthscale_interval=DEFAULT_LENGTHSCALE_INTERVAL,
        **kwargs,
    ):
        X_torch, Y_torch = self.get_XY(state)

        self.likelihood = GaussianLikelihood(noise_constraint=Interval(*noise_interval))
        covar_module = ScaleKernel(
            MaternKernel(nu=2.5, ard_num_dims=state.dim, lengthscale_constraint=Interval(*lengthscale_interval))
        )
        self.mean_module = mean_module
        self.model = SingleTaskGP(
            X_torch,
            Y_torch,
            mean_module=self.mean_module,
            covar_module=covar_module,
            likelihood=self.likelihood,
            **kwargs,
        )

        self.update_state_dicts()

    def fit(self, state: State, approx_mll=False, **kwargs):
        self.init_GP_model(state, self.mean_module, **kwargs)

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
        return {"mean": mean, "std": std}


class MultiFidelityGPSurrogate(BaseGPSurrogate):
    def init(self, state, noise_interval=DEFAULT_NOISE_INTERVAL, lengthscale_interval=DEFAULT_LENGTHSCALE_INTERVAL):
        X_torch, Y_torch = self.get_XY(state)

        self.likelihood = GaussianLikelihood(noise_constraint=Interval(*noise_interval))

        self.model = ModifiedSingleTaskMultiFidelityGP(
            X_torch, Y_torch, likelihood=self.likelihood, outcome_transform=None
        )

        self.update_state_dicts()

    def fit(self, state: State, approx_mll=False, **kwargs):
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
        return {fid: {"mean": mean[:, fid], "std": std[:, fid]} for fid in range(state.fidelity_domain.num_fidelities)}


##################################################
################ Ensemble Models #################
##################################################


class BasePyTorchModel(nn.Module, ABC):
    @property
    @abstractmethod
    def optimiser(self):
        pass

    @property
    @abstractmethod
    def scheduler(self):
        pass

    @staticmethod
    @abstractmethod
    def from_state_dict(state_dict):
        """
        Creates a instance of BasePyTorchModel from a given state_dict
        """
        pass


class EnsemblePyTorchModel(EnsembleModel):
    model_base_class: type[BasePyTorchModel]
    models: List[type[BasePyTorchModel]]
    starting_state_dicts: Dict
    _num_outputs: int
    ensemble_size: int
    training_epochs: int
    batch_size: int
    train_test_split: List[float]
    loss: type[nn.Module]
    reset_before_training: bool

    def __init__(
        self,
        ensemble_size: int,
        model_base_class: type[BasePyTorchModel],
        ensemble_state_dicts: Dict,
        training_epochs: int,
        batch_size: int,
        train_test_split: List[float],
        loss: type[nn.Module],
        reset_before_training: bool,
    ):
        super().__init__()
        self.model_base_class = model_base_class
        if ensemble_state_dicts is None:
            self.models = [self.model_base_class() for _ in range(ensemble_size)]
        else:
            assert ensemble_size == len(ensemble_state_dicts)
            self.models = [self.model_base_class.from_state_dict(sd) for _, sd in ensemble_state_dicts.items()]

        self.starting_state_dicts = {}
        for i, m in enumerate(self.models):
            self.starting_state_dicts[f"model_{i}_state_dict"] = copy.copy(m.state_dict(keep_vars=True))

        self.ensemble_size = ensemble_size
        self._num_outputs = 1

        self.training_epochs = training_epochs
        self.batch_size = batch_size
        self.train_test_split = train_test_split
        self.loss = loss
        self.reset_before_training = reset_before_training

    def get_state_dicts(self):
        state_dicts = {}
        for i, m in enumerate(self.models):
            state_dicts[f"model_{i}_state_dict"] = copy.copy(m.state_dict(keep_vars=True))
        return state_dicts

    def update_state_dicts(self, state_dicts):
        self.starting_state_dicts = state_dicts
        self.reset_models()

    def reset_models(self):
        self.models = [self.model_base_class.from_state_dict(sd) for _, sd in self.starting_state_dicts.items()]

    def fit(self, X: Tensor, y: Tensor) -> None:
        if self.reset_before_training:
            self.reset_models()

        dataset = TensorDataset(X, y)

        for m in self.models:
            model = m.model
            optimiser = m.optimiser
            scheduler = m.scheduler

            model.train()
            train_dataset, _ = random_split(dataset, self.train_test_split)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            for _ in range(self.training_epochs):
                for X_train, y_train in train_loader:
                    y_pred = model(X_train)
                    optimiser.zero_grad()
                    tloss = self.loss(y_pred, y_train)
                    tloss.backward()
                    optimiser.step()

                scheduler.step()

            model.eval()

    def forward(self, X: Tensor) -> Tensor:
        N, q, D = X.shape  # Samples, q-batch, Feature
        X_flat = X.reshape(N * q, D)

        outputs = [model.model(X_flat).reshape(N, q, -1) for model in self.models]

        return torch.stack(outputs, dim=1)  # (N, num_models, q, out_dim)


class BaseEnsemblePyTorchSurrogate(BaseSurrogate, ABC):
    """
    Abstract base class for all ensemble of PyTorch model surrogates in mille-feuille.
    """

    def __init__(
        self,
        ensemble_size: int,
        model_base_class: type[BasePyTorchModel],
        training_epochs: int,
        batch_size: int,
        pretrained_file: str | None = None,
        train_test_split: List[float] | None = None,
        loss: type[nn.Module] | None = None,
        reset_before_training: bool = True,
    ):
        self.model_base_class = model_base_class
        self.ensemble_size = ensemble_size

        if train_test_split is None:
            train_test_split = [0.7, 0.3]

        if loss is None:
            loss = nn.MSELoss()

        if pretrained_file is None:
            ensemble_state_dicts = None
        else:
            checkpoint = torch.load(pretrained_file, weights_only=False, map_location=device)
            ensemble_state_dicts = {}
            for k, v in checkpoint.items():
                ensemble_state_dicts[k] = v

        self.model = EnsemblePyTorchModel(
            ensemble_size,
            model_base_class,
            ensemble_state_dicts,
            training_epochs,
            batch_size,
            train_test_split,
            loss,
            reset_before_training,
        )
        self.ensemble_state_dicts = self.model.get_state_dicts()

    def eval(self):
        """
        Set internal models to evaluation mode.
        """
        for model in self.model.models:
            model.eval()

    def save(self, filepath: str):
        """
        Save trained models' state dicts to disk.
        """
        self.ensemble_state_dicts = self.model.get_state_dicts()
        torch.save(self.ensemble_state_dicts, filepath)

    def load(self, filepath: str, eval=True):
        """
        Load saved models' state dicts.
        """
        checkpoint = torch.load(filepath, weights_only=False, map_location=device)
        self.ensemble_state_dicts = {}
        for k, v in checkpoint.items():
            self.ensemble_state_dicts[k] = v

        self.model.update_state_dicts(self.ensemble_state_dicts)

        if eval:
            self.eval()


class SingleFidelityEnsembleSurrogate(BaseEnsemblePyTorchSurrogate):
    def fit(self, state: State):
        X_torch, Y_torch = self.get_XY(state)

        self.model.fit(X_torch, Y_torch)

    def predict(self, state: State, Xs):
        Xs_unit = state.transform_X(Xs)
        test_X = torch.tensor(Xs_unit, dtype=dtype, device=device)
        with torch.no_grad():
            post = self.model.posterior(test_X.unsqueeze(1))
            mean = post.mean.cpu().numpy().reshape(-1, 1)
            var = post.variance.cpu().numpy().reshape(-1, 1)
            std = np.sqrt(var)
        mean, std = state.inverse_transform_Y(mean, std)
        return {"mean": mean, "std": std}
