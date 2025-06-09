import math
from typing import Any, Optional, Union

import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.exceptions.errors import UnsupportedError
from botorch.models.gpytorch import GPyTorchModel, MultiTaskGPyTorchModel
from botorch.models.model import FantasizeMixin
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform, Standardize
from botorch.models.utils.gpytorch_modules import (
    get_covar_module_with_dim_scaled_prior,
    get_gaussian_likelihood_with_lognormal_prior,
    MIN_INFERRED_NOISE_LEVEL,
)
from botorch.posteriors.multitask import MultitaskGPPosterior
from botorch.utils.datasets import MultiTaskDataset, SupervisedDataset
from botorch.utils.types import _DefaultType, DEFAULT
from gpytorch.constraints import GreaterThan
from gpytorch.distributions.multitask_multivariate_normal import (
    MultitaskMultivariateNormal,
)
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.kernels.index_kernel import IndexKernel
from gpytorch.kernels.multitask_kernel import MultitaskKernel
from gpytorch.likelihoods.gaussian_likelihood import FixedNoiseGaussianLikelihood
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.likelihoods.multitask_gaussian_likelihood import (
    MultitaskGaussianLikelihood,
)
from gpytorch.means import MultitaskMean
from gpytorch.means.constant_mean import ConstantMean
from gpytorch.models.exact_gp import ExactGP
from gpytorch.module import Module
from gpytorch.priors.lkj_prior import LKJCovariancePrior
from gpytorch.priors.prior import Prior
from gpytorch.priors.smoothed_box_prior import SmoothedBoxPrior
from gpytorch.priors.torch_priors import GammaPrior, LogNormalPrior
from gpytorch.settings import detach_test_caches
from gpytorch.utils.errors import CachingError
from gpytorch.utils.memoize import cached, pop_from_cache
from linear_operator.operators import (
    BatchRepeatLinearOperator,
    CatLinearOperator,
    DiagLinearOperator,
    KroneckerProductDiagLinearOperator,
    KroneckerProductLinearOperator,
    RootLinearOperator,
    to_linear_operator,
)
from torch import Tensor


def get_task_value_remapping(
    task_values: Tensor, dtype: torch.dtype
) -> Optional[Tensor]:
    """Construct an mapping of discrete task values to contiguous int-valued floats.

    Args:
        task_values: A sorted long-valued tensor of task values.
        dtype: The dtype of the model inputs (e.g. `X`), which the new
            task values should have mapped to (e.g. float, double).

    Returns:
        A tensor of shape `task_values.max() + 1` that maps task values
        to new task values. The indexing operation `mapper[task_value]`
        will produce a tensor of new task values, of the same shape as
        the original. The elements of the `mapper` tensor that do not
        appear in the original `task_values` are mapped to `nan`. The
        return value will be `None`, when the task values are contiguous
        integers starting from zero.
    """
    task_range = torch.arange(
        len(task_values), dtype=task_values.dtype, device=task_values.device
    )
    mapper = None
    if not torch.equal(task_values, task_range):
        # Create a tensor that maps task values to new task values.
        # The number of tasks should be small, so this should be quite efficient.
        mapper = torch.full(
            (int(task_values.max().item()) + 1,),
            float("nan"),
            dtype=dtype,
            device=task_values.device,
        )
        mapper[task_values] = task_range.to(dtype=dtype)
    return mapper


class ModifiedMultiTaskGP(ExactGP, MultiTaskGPyTorchModel, FantasizeMixin):
    r"""Multi-Task exact GP model using an ICM (intrinsic co-regionalization model)
    kernel. See [Bonilla2007MTGP]_ and [Swersky2013MTBO]_ for a reference on the
    model and its use in Bayesian optimization.

    The model can be single-output or multi-output, determined by the `output_tasks`.
    This model uses relatively strong priors on the base Kernel hyperparameters, which
    work best when covariates are normalized to the unit cube and outcomes are
    standardized (zero mean, unit variance) - this standardization should be applied in
    a stratified fashion at the level of the tasks, rather than across all data points.

    If the `train_Yvar` is None, this model infers the noise level. If you have
    known observation noise, you can set `train_Yvar` to a tensor containing
    the noise variance measurements. WARNING: This currently does not support
    different noise levels for the different tasks.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        task_feature: int,
        train_Yvar: Optional[Tensor] = None,
        mean_module: Optional[Module] = None,
        covar_module: Optional[Module] = None,
        covar_base_module: Optional[Module] = None,
        likelihood: Optional[Likelihood] = None,
        task_covar_prior: Optional[Prior] = None,
        output_tasks: Optional[list[int]] = None,
        rank: Optional[int] = None,
        all_tasks: Optional[list[int]] = None,
        outcome_transform: Optional[Union[OutcomeTransform, _DefaultType]] = DEFAULT,
        input_transform: Optional[InputTransform] = None,
    ) -> None:
        r"""Multi-Task GP model using an ICM kernel.

        Args:
            train_X: A `n x (d + 1)` or `b x n x (d + 1)` (batch mode) tensor
                of training data. One of the columns should contain the task
                features (see `task_feature` argument).
            train_Y: A `n x 1` or `b x n x 1` (batch mode) tensor of training
                observations.
            task_feature: The index of the task feature (`-d <= task_feature <= d`).
            train_Yvar: An optional `n` or `b x n` (batch mode) tensor of observed
                measurement noise. If None, we infer the noise.
                Note that the inferred noise is common across all tasks.
            mean_module: The mean function to be used. Defaults to `ConstantMean`.
            covar_module: The module for computing the covariance matrix between
                the non-task features. Defaults to `RBFKernel`.
            likelihood: A likelihood. The default is selected based on `train_Yvar`.
                If `train_Yvar` is None, a standard `GaussianLikelihood` with inferred
                noise level is used. Otherwise, a FixedNoiseGaussianLikelihood is used.
            output_tasks: A list of task indices for which to compute model
                outputs for. If omitted, return outputs for all task indices.
            rank: The rank to be used for the index kernel. If omitted, use a
                full rank (i.e. number of tasks) kernel.
            task_covar_prior : A Prior on the task covariance matrix. Must operate
                on p.s.d. matrices. A common prior for this is the `LKJ` prior.
            all_tasks: By default, multi-task GPs infer the list of all tasks from
                the task features in `train_X`. This is an experimental feature that
                enables creation of multi-task GPs with tasks that don't appear in the
                training data. Note that when a task is not observed, the corresponding
                task covariance will heavily depend on random initialization and may
                behave unexpectedly.
            outcome_transform: An outcome transform that is applied to the
                training data during instantiation and to the posterior during
                inference (that is, the `Posterior` obtained by calling
                `.posterior` on the model will be on the original scale). We use a
                `Standardize` transform if no `outcome_transform` is specified.
                Pass down `None` to use no outcome transform. NOTE: Standardization
                should be applied in a stratified fashion, separately for each task.
            input_transform: An input transform that is applied in the model's
                forward pass.

        Example:
            >>> X1, X2 = torch.rand(10, 2), torch.rand(20, 2)
            >>> i1, i2 = torch.zeros(10, 1), torch.ones(20, 1)
            >>> train_X = torch.cat([
            >>>     torch.cat([X1, i1], -1), torch.cat([X2, i2], -1),
            >>> ])
            >>> train_Y = torch.cat(f1(X1), f2(X2)).unsqueeze(-1)
            >>> model = MultiTaskGP(train_X, train_Y, task_feature=-1)
        """
        with torch.no_grad():
            transformed_X = self.transform_inputs(
                X=train_X, input_transform=input_transform
            )
        self._validate_tensor_args(X=transformed_X, Y=train_Y, Yvar=train_Yvar)
        (
            all_tasks_inferred,
            task_feature,
            self.num_non_task_features,
        ) = self.get_all_tasks(transformed_X, task_feature, output_tasks)
        if all_tasks is not None and not set(all_tasks_inferred).issubset(all_tasks):
            raise UnsupportedError(
                f"The provided {all_tasks=} does not contain all the task features "
                f"inferred from the training data {all_tasks_inferred=}. "
                "This is not allowed as it will lead to errors during model training."
            )
        all_tasks = all_tasks or all_tasks_inferred
        self.num_tasks = len(all_tasks)
        if outcome_transform == DEFAULT:
            outcome_transform = Standardize(m=1, batch_shape=train_X.shape[:-2])
        if outcome_transform is not None:
            train_Y, train_Yvar = outcome_transform(Y=train_Y, Yvar=train_Yvar)

        # squeeze output dim
        train_Y = train_Y.squeeze(-1)
        if output_tasks is None:
            output_tasks = all_tasks
        else:
            if set(output_tasks) - set(all_tasks):
                raise RuntimeError("All output tasks must be present in input data.")
        self._output_tasks = output_tasks
        self._num_outputs = len(output_tasks)

        # TODO (T41270962): Support task-specific noise levels in likelihood
        if likelihood is None:
            if train_Yvar is None:
                likelihood = get_gaussian_likelihood_with_lognormal_prior()
            else:
                likelihood = FixedNoiseGaussianLikelihood(noise=train_Yvar.squeeze(-1))

        # construct indexer to be used in forward
        self._task_feature = task_feature
        self._base_idxr = torch.arange(self.num_non_task_features)
        self._base_idxr[task_feature:] += 1  # exclude task feature

        super().__init__(
            train_inputs=train_X, train_targets=train_Y, likelihood=likelihood
        )
        self.mean_module = mean_module or ConstantMean()
        if covar_module is None:
            self.covar_module = get_covar_module_with_dim_scaled_prior(
                ard_num_dims=self.num_non_task_features
            )
        else:
            self.covar_module = covar_module

        if covar_base_module is None:
            self.covar_base_module = get_covar_module_with_dim_scaled_prior(
                ard_num_dims=self.num_non_task_features
            )
        else:
            self.covar_base_module = covar_base_module

        self._rank = rank if rank is not None else self.num_tasks
        self.task_covar_module = IndexKernel(
            num_tasks=self.num_tasks, rank=self._rank, prior=task_covar_prior
        )
        task_mapper = get_task_value_remapping(
            task_values=torch.tensor(
                all_tasks, dtype=torch.long, device=train_X.device
            ),
            dtype=train_X.dtype,
        )
        self.register_buffer("_task_mapper", task_mapper)
        self._expected_task_values = set(all_tasks)
        if input_transform is not None:
            self.input_transform = input_transform
        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        self.to(train_X)

    def _split_inputs(self, x: Tensor) -> tuple[Tensor, Tensor]:
        r"""Extracts base features and task indices from input data.

        Args:
            x: The full input tensor with trailing dimension of size `d + 1`.
                Should be of float/double data type.

        Returns:
            2-element tuple containing

            - A `q x d` or `b x q x d` (batch mode) tensor with trailing
            dimension made up of the `d` non-task-index columns of `x`, arranged
            in the order as specified by the indexer generated during model
            instantiation.
            - A `q` or `b x q` (batch mode) tensor of long data type containing
            the task indices.
        """
        batch_shape, d = x.shape[:-2], x.shape[-1]
        x_basic = x[..., self._base_idxr].view(batch_shape + torch.Size([-1, d - 1]))
        task_idcs = (
            x[..., self._task_feature]
            .view(batch_shape + torch.Size([-1, 1]))
            .to(dtype=torch.long)
        )
        task_idcs = self._map_tasks(task_values=task_idcs)
        return x_basic, task_idcs

    def forward(self, x: Tensor) -> MultivariateNormal:
        if self.training:
            x = self.transform_inputs(x)
        x_basic, task_idcs = self._split_inputs(x)
        # Compute base mean and covariance
        mean_x = self.mean_module(x_basic)
        covar_base_x = self.covar_base_module(x_basic)
        # Compute task covariances
        covar_x = self.covar_module(x_basic)
        covar_i = self.task_covar_module(task_idcs)
        # Combine in an ICM fashion
        covar = covar_base_x + covar_x.mul(covar_i)
        return MultivariateNormal(mean_x, covar)

    @classmethod
    def get_all_tasks(
        cls,
        train_X: Tensor,
        task_feature: int,
        output_tasks: Optional[list[int]] = None,
    ) -> tuple[list[int], int, int]:
        if train_X.ndim != 2:
            # Currently, batch mode MTGPs are blocked upstream in GPyTorch
            raise ValueError(f"Unsupported shape {train_X.shape} for train_X.")

        d = train_X.shape[-1] - 1
        if not (-d <= task_feature <= d):
            raise ValueError(f"Must have that -{d} <= task_feature <= {d}")
        task_feature = task_feature % (d + 1)
        all_tasks = (
            train_X[..., task_feature].unique(sorted=True).to(dtype=torch.long).tolist()
        )
        return all_tasks, task_feature, d

    @classmethod
    def construct_inputs(
        cls,
        training_data: Union[SupervisedDataset, MultiTaskDataset],
        task_feature: int,
        output_tasks: Optional[list[int]] = None,
        task_covar_prior: Optional[Prior] = None,
        prior_config: Optional[dict] = None,
        rank: Optional[int] = None,
    ) -> dict[str, Any]:
        r"""Construct `Model` keyword arguments from a dataset and other args.

        Args:
            training_data: A `SupervisedDataset` or a `MultiTaskDataset`.
            task_feature: Column index of embedded task indicator features.
            output_tasks: A list of task indices for which to compute model
                outputs for. If omitted, return outputs for all task indices.
            task_covar_prior: A GPyTorch `Prior` object to use as prior on
                the cross-task covariance matrix,
            prior_config: Configuration for inter-task covariance prior.
                Should only be used if `task_covar_prior` is not passed directly. Must
                contain `use_LKJ_prior` indicator and should contain float value `eta`.
            rank: The rank of the cross-task covariance matrix.
        """
        if task_covar_prior is not None and prior_config is not None:
            raise ValueError(
                "Only one of `task_covar_prior` and `prior_config` arguments expected."
            )

        if prior_config is not None:
            if not prior_config.get("use_LKJ_prior"):
                raise ValueError("Currently only config for LKJ prior is supported.")

            num_tasks = training_data.X[task_feature].unique().numel()
            sd_prior = GammaPrior(1.0, 0.15)
            sd_prior._event_shape = torch.Size([num_tasks])
            eta = prior_config.get("eta", 0.5)
            if not isinstance(eta, float) and not isinstance(eta, int):
                raise ValueError(f"eta must be a real number, your eta was {eta}.")
            task_covar_prior = LKJCovariancePrior(num_tasks, eta, sd_prior)

        # Call Model.construct_inputs to parse training data
        base_inputs = super().construct_inputs(training_data=training_data)
        if (
            isinstance(training_data, MultiTaskDataset)
            # If task features are included in the data, all tasks will have
            # some observations and they may have different task features.
            and training_data.task_feature_index is None
        ):
            all_tasks = list(range(len(training_data.datasets)))
            base_inputs["all_tasks"] = all_tasks
        if task_covar_prior is not None:
            base_inputs["task_covar_prior"] = task_covar_prior
        if rank is not None:
            base_inputs["rank"] = rank
        base_inputs["task_feature"] = task_feature
        base_inputs["output_tasks"] = output_tasks
        return base_inputs


from typing import Any

import torch
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.kernels.downsampling import DownsamplingKernel

from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
from botorch.utils.datasets import SupervisedDataset
from botorch.utils.types import _DefaultType, DEFAULT
from gpytorch.kernels.kernel import ProductKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.priors.torch_priors import GammaPrior
from torch import Tensor

class ModifiedSingleTaskMultiFidelityGP(SingleTaskGP):
    r"""A single task multi-fidelity GP model.

    A SingleTaskGP model using a DownsamplingKernel for the data fidelity
    parameter (if present) and an ExponentialDecayKernel for the iteration
    fidelity parameter (if present).

    This kernel is described in [Wu2019mf]_.

    Example:
        >>> train_X = torch.rand(20, 4)
        >>> train_Y = train_X.pow(2).sum(dim=-1, keepdim=True)
        >>> model = SingleTaskMultiFidelityGP(train_X, train_Y, data_fidelities=[3])
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
            transformed_X = self.transform_inputs(
                X=train_X, input_transform=input_transform
            )

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

    non_active_dims = set([dim-1])
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
            active_dims=[dim-1],
        )
    )

    kernel = ProductKernel(*kernels)

    base_kernel = get_covar_module_with_dim_scaled_prior(
            ard_num_dims=len(active_dimsX),
            batch_shape=aug_batch_shape,
            active_dims=active_dimsX,
        )

    kernel = base_kernel + kernel

    covar_module = ScaleKernel(
        kernel, batch_shape=aug_batch_shape, outputscale_prior=GammaPrior(2.0, 0.15)
    )

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