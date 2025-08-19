from typing import Any

import torch
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.kernels.downsampling import DownsamplingKernel
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.models.utils.gpytorch_modules import (
    get_covar_module_with_dim_scaled_prior,
)
from botorch.utils.datasets import SupervisedDataset
from botorch.utils.types import DEFAULT, _DefaultType
from gpytorch.kernels.kernel import ProductKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.priors.torch_priors import GammaPrior
from torch import Tensor


class ModifiedSingleTaskMultiFidelityGP(SingleTaskGP):
    r"""A single task multi-fidelity GP model.

    A SingleTaskGP model using a DownsamplingKernel for the fidelity
    parameter.

    This kernel is described in [Wu2019mf]_.

    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        likelihood: Likelihood | None = None,
        outcome_transform: OutcomeTransform | _DefaultType | None = DEFAULT,
        input_transform: InputTransform | None = None,
    ) -> None:
        r"""
        Args:
            train_X: A `batch_shape x n x (d + s)` tensor of training features,
                where `s` is the dimension of the fidelity parameters (=1).
            train_Y: A `batch_shape x n x m` tensor of training observations.
            train_Yvar: An optional `batch_shape x n x m` tensor of observed
                measurement noise.
            likelihood: A likelihood. If omitted, use a standard GaussianLikelihood
                with inferred noise level.
            outcome_transform: An outcome transform that is applied to the
                training data during instantiation and to the posterior during
                inference (that is, the `Posterior` obtained by calling
                `.posterior` on the model will be on the original scale). We use a
                `Standardize` transform if no `outcome_transform` is specified.
                Pass down `None` to use no outcome transform.
            input_transform: An input transform that is applied in the model's
                    forward pass.
        """
        self._init_args = {
            "outcome_transform": outcome_transform,
        }

        with torch.no_grad():
            transformed_X = self.transform_inputs(X=train_X, input_transform=input_transform)

        self._set_dimensions(train_X=transformed_X, train_Y=train_Y)
        covar_module, subset_batch_dict = _setup_multifidelity_covar_module(
            dim=transformed_X.size(-1),
            aug_batch_shape=self._aug_batch_shape,
        )
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            likelihood=likelihood,
            covar_module=covar_module,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )
        # Used for subsetting along the output dimension. See Model.subset_output.
        self._subset_batch_dict = {
            "mean_module.raw_constant": -1,
            "covar_module.raw_outputscale": -1,
            **subset_batch_dict,
        }
        if train_Yvar is None:
            self._subset_batch_dict["likelihood.noise_covar.raw_noise"] = -2
        self.to(train_X)

    @classmethod
    def construct_inputs(
        cls,
        training_data: SupervisedDataset,
        fidelity_features: list[int],
    ) -> dict[str, Any]:
        r"""Construct `Model` keyword arguments from a dict of `SupervisedDataset`.

        Args:
            training_data: Dictionary of `SupervisedDataset`.
            fidelity_features: Index of fidelity parameter as input columns.
        """
        inputs = super().construct_inputs(training_data=training_data)
        inputs["data_fidelities"] = fidelity_features
        return inputs


def _setup_multifidelity_covar_module(
    dim: int,
    aug_batch_shape: torch.Size,
) -> tuple[ScaleKernel, dict]:
    """Helper function to get the covariance module and associated subset_batch_dict
    for the multifidelity setting.

    Args:
        dim: The dimensionality of the training data.
        aug_batch_shape: The output-augmented batch shape as defined in
            `BatchedMultiOutputGPyTorchModel`.

    Returns:
        The covariance module and subset_batch_dict.
    """

    kernels = []

    non_active_dims = set([dim - 1])
    active_dimsX = sorted(set(range(dim)) - non_active_dims)
    kernels.append(
        get_covar_module_with_dim_scaled_prior(
            ard_num_dims=len(active_dimsX),
            batch_shape=aug_batch_shape,
            active_dims=active_dimsX,
        )
    )

    kernels.append(
        DownsamplingKernel(
            batch_shape=aug_batch_shape,
            offset_prior=GammaPrior(3.0, 6.0),
            power_prior=GammaPrior(3.0, 6.0),
            active_dims=[dim - 1],
        )
    )

    kernel = ProductKernel(*kernels)

    base_kernel = get_covar_module_with_dim_scaled_prior(
        ard_num_dims=len(active_dimsX),
        batch_shape=aug_batch_shape,
        active_dims=active_dimsX,
    )

    kernel = base_kernel + kernel

    covar_module = ScaleKernel(kernel, batch_shape=aug_batch_shape, outputscale_prior=GammaPrior(2.0, 0.15))

    key_prefix = "covar_module.base_kernel.kernels"

    subset_batch_dict = {
        f"{key_prefix}.0.raw_lengthscale": -3,
    }

    start_idx = 1
    for i in range(start_idx, 1 + start_idx):
        subset_batch_dict.update(
            {
                f"{key_prefix}.{i}.raw_power": -2,
                f"{key_prefix}.{i}.raw_offset": -2,
            }
        )

    return covar_module, subset_batch_dict
