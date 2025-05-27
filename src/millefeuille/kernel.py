"""
Defines number of kernels beyond the standards in BOtorch and GPytorch
"""

from botorch.models.gp_regression import SingleTaskGP
from botorch.models.kernels.categorical import CategoricalKernel
from botorch.utils.transforms import normalize_indices

from gpytorch.constraints import GreaterThan, Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.kernels.kernel import Kernel

from linear_operator.operators import DiagLinearOperator, InterpolatedLinearOperator

import torch

class DiagonalIndexKernel(Kernel):
    r"""
    Based on IndexKernel

    A kernel for discrete indices. Kernel is defined by a lookup table.

    .. math::

        \begin{equation}
            k(i, j) = \text{diag}(\mathbf v)_{i, j}
        \end{equation}

    where :math:`\mathbf v` is a  non-negative vector.
    These parameters are learned.

    Args:
        num_tasks (int):
            Total number of indices.
        task_var (int):
            v[task_var] = 1.0, v[:task_var] = 0.0
        batch_shape (torch.Size, optional):
            Set if the MultitaskKernel is operating on batches of data (and you want different
            parameters for each batch)

    Attributes:
        raw_var:
            The element-wise `Softplus <https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html>`_
            of the :math:`\mathbf v` vector (assuming the default `var_constraint`).
    """

    def __init__(
        self,
        num_tasks: int,
        task_var : int,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.static_var = torch.zeros(task_var+1)
        self.static_var[-1] = 1.0

        learnable_var = torch.nn.Parameter(torch.randn(*self.batch_shape, num_tasks-task_var-1))
        self.register_parameter(name="raw_var", parameter=learnable_var)

        self.register_constraint("raw_var", Interval(0.0,1.0))

    @property
    def var(self):
        return self.raw_var_constraint.transform(self.raw_var)

    @var.setter
    def var(self, value):
        self._set_var(value)

    def _set_var(self, value):
        self.initialize(raw_var=self.raw_var_constraint.inverse_transform(value))

    def _eval_covar_matrix(self):
        diag = torch.cat([self.static_var,self.var])
        return torch.diag_embed(diag)

    @property
    def covar_matrix(self):
        diag = torch.cat([self.static_var,self.var])
        return DiagLinearOperator(diag)

    def forward(self, i1, i2, **params):

        i1, i2 = i1.long(), i2.long()
        covar_matrix = self._eval_covar_matrix()
        batch_shape = torch.broadcast_shapes(i1.shape[:-2], i2.shape[:-2], self.batch_shape)

        res = InterpolatedLinearOperator(
            base_linear_op=covar_matrix,
            left_interp_indices=i1.expand(batch_shape + i1.shape[-2:]),
            right_interp_indices=i2.expand(batch_shape + i2.shape[-2:]),
        )
        return res

class Hierarchical_MultiFidelity_MaternGP(SingleTaskGP):
    """

    """

    def __init__(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        num_fidelities : int,
        nu: float,
        lengthscale_interval: list[float],
        train_Yvar: torch.Tensor | None = None,
        likelihood: Likelihood | None = None,
    ) -> None:
        r"""A single-task exact GP model supporting categorical parameters.

        Args:
            train_X: A `batch_shape x n x d` tensor of training features.
            train_Y: A `batch_shape x n x m` tensor of training observations.
            num_fidelities : number of fidelities considered.
            nu : nu used for Matern kernel.
            lengthscale_interval : lengthscale constraints for Matern kernel.
            train_Yvar: An optional `batch_shape x n x m` tensor of observed
                measurement noise.
            likelihood: A likelihood. If omitted, use a standard
                GaussianLikelihood with inferred noise level.
        """

        d = train_X.shape[-1]

        # Fidelity variable append on X
        cat_dims = [d-1]
        self._ignore_X_dims_scaling_check = cat_dims
        _, aug_batch_shape = self.get_batch_dimensions(train_X=train_X, train_Y=train_Y)
        
        cat_dims = normalize_indices(indices=cat_dims, d=d)
        ord_dims = sorted(set(range(d)) - set(cat_dims))

        # Base kernel
        covar_module = ScaleKernel(
            MaternKernel(
                nu=nu,
                batch_shape=aug_batch_shape,
                ard_num_dims=len(ord_dims),
                active_dims=ord_dims,
                lengthscale_constraint=Interval(*lengthscale_interval),
            )             
            * DiagonalIndexKernel(
                num_tasks=num_fidelities,
                task_var=0,
                batch_shape=aug_batch_shape,
                ard_num_dims=len(cat_dims),
                active_dims=cat_dims,
            )
        )

        # Add discrepancy kernels
        for s in range(1,num_fidelities):

            fid_kernel = ScaleKernel(
                MaternKernel(
                    nu=nu,
                    batch_shape=aug_batch_shape,
                    ard_num_dims=len(ord_dims),
                    active_dims=ord_dims,
                    lengthscale_constraint=Interval(*lengthscale_interval),
                )
                * DiagonalIndexKernel(
                    num_tasks=num_fidelities,
                    task_var=s,
                    batch_shape=aug_batch_shape,
                    ard_num_dims=len(cat_dims),
                    active_dims=cat_dims,
                )
            )

            covar_module += fid_kernel

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            likelihood=likelihood,
            covar_module=covar_module
        )

class MF_Matern_SingleTaskGP(SingleTaskGP):
    r"""
    Edited version of MixedSingleTaskGP

        > Categorical variable assumed to be last dimension
        > Matern kernel used for continuous variables
        > Categorical kernel removed some sum kernel

    Original doc string:
    
    A single-task exact GP model for mixed search spaces.

    This model is similar to `SingleTaskGP`, but supports mixed search spaces,
    which combine discrete and continuous features, as well as solely discrete
    spaces. It uses a kernel that combines a CategoricalKernel (based on
    Hamming distances) and a regular kernel into a kernel of the form

        K((x1, c1), (x2, c2)) =
            K_cont_1(x1, x2) + 
            K_cont_2(x1, x2) * K_cat_2(c1, c2)

    where `xi` and `ci` are the continuous and categorical features of the
    input, respectively. The suffix `_i` indicates that we fit different
    lengthscales for the kernels in the sum and product terms.

    Since this model does not provide gradients for the categorical features,
    optimization of the acquisition function will need to be performed in
    a mixed fashion, i.e., treating the categorical features properly as
    discrete optimization variables. We recommend using `optimize_acqf_mixed.`

    """

    def __init__(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        nu: float,
        lengthscale_interval: list[float],
        train_Yvar: torch.Tensor | None = None,
        likelihood: Likelihood | None = None,
    ) -> None:
        r"""A single-task exact GP model supporting categorical parameters.

        Args:
            train_X: A `batch_shape x n x d` tensor of training features.
            train_Y: A `batch_shape x n x m` tensor of training observations.
            nu : nu used for Matern kernel
            lengthscale_interval : lengthscale constraints for Matern kernel
            train_Yvar: An optional `batch_shape x n x m` tensor of observed
                measurement noise.
            likelihood: A likelihood. If omitted, use a standard
                GaussianLikelihood with inferred noise level.
        """

        d = train_X.shape[-1]

        # Fidelity variable append on X
        cat_dims = [d-1]
        self._ignore_X_dims_scaling_check = cat_dims
        _, aug_batch_shape = self.get_batch_dimensions(train_X=train_X, train_Y=train_Y)
        
        cat_dims = normalize_indices(indices=cat_dims, d=d)
        ord_dims = sorted(set(range(d)) - set(cat_dims))

        joint_kernel = ScaleKernel(
            MaternKernel(
                nu=nu,
                batch_shape=aug_batch_shape,
                ard_num_dims=len(ord_dims),
                active_dims=ord_dims,
                lengthscale_constraint=Interval(*lengthscale_interval),
            )
        )

        prod_kernel = ScaleKernel(
            MaternKernel(
                nu=nu,
                batch_shape=aug_batch_shape,
                ard_num_dims=len(ord_dims),
                active_dims=ord_dims,
                lengthscale_constraint=Interval(*lengthscale_interval),
            )
            * CategoricalKernel(
                batch_shape=aug_batch_shape,
                ard_num_dims=len(cat_dims),
                active_dims=cat_dims,
                lengthscale_constraint=GreaterThan(1e-06),
            )
        )
        covar_module = joint_kernel + prod_kernel

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            likelihood=likelihood,
            covar_module=covar_module,
            outcome_transfrom=None
        )