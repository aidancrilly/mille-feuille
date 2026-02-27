from dataclasses import dataclass

import numpy as np
from botorch.acquisition.utils import project_to_target_fidelity

"""
Define the domains of the problem
"""


@dataclass
class InputDomain:
    """Defines an input domain which can have both discrete and continuous dimensions.

    The domain is internally scaled to the unit hypercube ``[0, 1]^dim`` for
    surrogate training.

    Parameters:
        dim: Number of input dimensions.
        b_low: Lower bounds for each dimension, shape ``(dim,)``.
        b_up: Upper bounds for each dimension, shape ``(dim,)``.
        steps: Step size for discrete dimensions; use ``0.0`` for continuous
            dimensions, shape ``(dim,)``.

    Attributes:
        discrete_indices: Indices of discrete dimensions (where ``steps != 0``).
        discrete_dim: Number of discrete dimensions.
        discrete_bound: Number of discrete values per discrete dimension.

    Example:
        >>> import numpy as np
        >>> from millefeuille.domain import InputDomain
        >>> domain = InputDomain(
        ...     dim=2,
        ...     b_low=np.array([0.0, 0.0]),
        ...     b_up=np.array([1.0, 10.0]),
        ...     steps=np.array([0.0, 1.0]),
        ... )
    """

    dim: int
    b_low: np.ndarray
    b_up: np.ndarray
    steps: np.ndarray

    def __post_init__(self):
        self.discrete_indices = [i for i in range(self.dim) if self.steps[i] != 0.0]
        self.discrete_dim = len(self.discrete_indices)
        self.discrete_bound = [int((self.b_up[i] - self.b_low[i]) / self.steps[i]) for i in self.discrete_indices]

    def get_bounds(self):
        """Return the unit-hypercube bounds as a ``(2, dim)`` array.

        Returns:
            np.ndarray: Array of shape ``(2, dim)`` where ``bounds[0]`` is all
            zeros and ``bounds[1]`` is all ones.
        """
        lb = np.zeros(self.dim)
        ub = np.ones(self.dim)
        bounds = np.stack([lb, ub])
        return bounds

    def transform(self, X):
        """Scale input samples from parameter space to the unit hypercube.

        Parameters:
            X: Input array of shape ``(N, dim)`` in parameter space.

        Returns:
            np.ndarray: Scaled array of shape ``(N, dim)`` in ``[0, 1]^dim``.
        """
        X_scaled = X.copy()
        # Transform to [0,1]^d
        for n in range(self.dim):
            X_scaled[:, n] = (X_scaled[:, n] - self.b_low[n]) / (self.b_up[n] - self.b_low[n])
        # Catch floating point errors
        X_scaled[np.isclose(X_scaled, 0.0)] = 0.0
        X_scaled[np.isclose(X_scaled, 1.0)] = 1.0
        return X_scaled

    def inverse_transform(self, X):
        """Map unit-hypercube samples back to parameter space.

        For discrete dimensions, values are rounded to the nearest grid point.

        Parameters:
            X: Scaled input array of shape ``(N, dim)`` in ``[0, 1]^dim``.

        Returns:
            np.ndarray: Array of shape ``(N, dim)`` in parameter space.
        """
        X_scaled = X.copy()
        # Scale back to parameter space
        for n in range(self.dim):
            X_scaled[:, n] = (self.b_up[n] - self.b_low[n]) * X_scaled[:, n] + self.b_low[n]
            # For discrete spaces, find nearest point
            if n in self.discrete_indices:
                X_scaled[:, n] = np.rint(X_scaled[:, n] / self.steps[n]) * self.steps[n]
        return X_scaled


@dataclass
class FidelityDomain:
    """Defines the fidelity space domain, which is discrete.

    Fidelities are indexed from ``0`` (lowest) to ``num_fidelities - 1``
    (highest / target).  Optional per-fidelity costs can be specified for
    cost-aware acquisition functions (e.g. :func:`~millefeuille.cost.generate_multifidelity_cost_model`).

    Parameters:
        num_fidelities: Total number of fidelity levels.
        costs: Per-fidelity cost values.  If ``None``, all fidelities are
            assigned a cost of ``1.0``.

    Attributes:
        minimal_fidelity: Index of the cheapest fidelity (always ``0``).
        target_fidelity: Index of the highest fidelity (``num_fidelities - 1``).
        fidelities: List ``[0, 1, ..., num_fidelities - 1]``.

    Example:
        >>> from millefeuille.domain import FidelityDomain
        >>> fd = FidelityDomain(num_fidelities=3, costs=[0.1, 0.5, 1.0])
    """

    num_fidelities: int
    costs: None | list = None

    def __post_init__(self):
        if self.costs is None:
            self.costs = [1.0 for _ in range(self.num_fidelities)]
        else:
            assert len(self.costs) == self.num_fidelities

        self.minimal_fidelity = 0
        self.target_fidelity = self.num_fidelities - 1
        self.fidelities = [i for i in range(self.num_fidelities)]

    def combine_with_input_domain(self, input_dim):
        """Compute fidelity metadata relative to an input domain of dimension *input_dim*.

        Populates :attr:`fidelity_weights`, :attr:`target_fidelities`, and
        :attr:`fidelity_features` which are required by BoTorch multi-fidelity
        acquisition functions.

        Parameters:
            input_dim: Number of input (non-fidelity) dimensions in the problem.
        """
        self.fidelity_weights = {input_dim: 1.0}
        self.target_fidelities = {input_dim: self.target_fidelity}
        self.fidelity_features = [{input_dim: f} for f in self.fidelities]

    def get_bounds(self):
        """Return a 1-D array containing ``[minimal_fidelity, target_fidelity]``.

        Returns:
            np.ndarray: Array of shape ``(2,)`` with the lower and upper fidelity
            indices, suitable for appending to input-space bounds.
        """
        bounds = np.array([self.minimal_fidelity, self.target_fidelity])
        return bounds

    def project(self, XSs):
        """Project candidate points to the target fidelity.

        Thin wrapper around
        :func:`botorch.acquisition.utils.project_to_target_fidelity`.

        Parameters:
            XSs: Tensor of shape ``(q, d + 1)`` where the last column is the
                fidelity index.

        Returns:
            torch.Tensor: Tensor with the fidelity column set to
            :attr:`target_fidelity`.
        """
        return project_to_target_fidelity(X=XSs, target_fidelities=self.target_fidelities)
