"""
Defines number of kernels beyond the standards in BOtorch and GPytorch
"""

from botorch.models.gp_regression import SingleTaskGP
from botorch.models.kernels.categorical import CategoricalKernel
from botorch.utils.transforms import normalize_indices
from gpytorch.constraints import GreaterThan,Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.likelihoods.likelihood import Likelihood
from torch import Tensor

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
        train_X: Tensor,
        train_Y: Tensor,
        nu: float,
        lengthscale_interval: list[float],
        train_Yvar: Tensor | None = None,
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
            covar_module=covar_module
        )