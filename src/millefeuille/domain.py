from dataclasses import dataclass

import numpy as np
from botorch.acquisition.utils import project_to_target_fidelity

"""
Define the domains of the problem
"""


@dataclass
class InputDomain:
    """Defines an input domain which can have both discrete and continuous dimensions.

    This class represents a mixed continuous-discrete optimization domain that is scaled
    to the unit hypercube [0, 1]^d. Continuous dimensions are specified with steps=0.0,
    while discrete dimensions have non-zero step sizes.

    Attributes:
        dim (int): Total number of dimensions in the domain.
        b_low (np.ndarray): Lower bounds of each dimension in real units (shape: (dim,)).
        b_up (np.ndarray): Upper bounds of each dimension in real units (shape: (dim,)).
        steps (np.ndarray): Step sizes for each dimension. Zero indicates continuous dimension,
            non-zero indicates discrete dimension with that step size (shape: (dim,)).
        discrete_indices (list): Indices of discrete dimensions (computed in __post_init__).
        discrete_dim (int): Number of discrete dimensions (computed in __post_init__).
        discrete_bound (list): Number of discrete levels for each discrete dimension
            (computed in __post_init__).
    """

    dim: int
    b_low: np.ndarray
    b_up: np.ndarray
    steps: np.ndarray

    def __post_init__(self):
        """Compute discrete dimension indices and bounds after initialization.

        Sets discrete_indices, discrete_dim, and discrete_bound attributes based on
        the steps array. Dimensions with steps[i] != 0.0 are treated as discrete.
        """
        self.discrete_indices = [i for i in range(self.dim) if self.steps[i] != 0.0]
        self.discrete_dim = len(self.discrete_indices)
        self.discrete_bound = [int((self.b_up[i] - self.b_low[i]) / self.steps[i]) for i in self.discrete_indices]

    def get_bounds(self):
        """Get the bounds of the normalized domain [0, 1]^d.

        Returns:
            np.ndarray: Shape (2, dim) array where first row is lower bounds (0.0)
                and second row is upper bounds (1.0).
        """
        lb = np.zeros(self.dim)
        ub = np.ones(self.dim)
        bounds = np.stack([lb, ub])
        return bounds
    
    @staticmethod
    def read_json(filepath: str) -> tuple["InputDomain", list[str]]:
        """Create an ``InputDomain`` from a JSON configuration file.

        The JSON file must contain a ``"params"`` object with keys
        ``"names"``, ``"lower_bounds"``, ``"upper_bounds"`` and ``"steps"``.

        Parameters:
            filepath: Path to the JSON file.

        Returns:
            A tuple ``(domain, X_names)`` where *domain* is the constructed
            ``InputDomain`` and *X_names* is the list of parameter names.
        """
        with open(filepath, "r") as f:
            cfg = json.load(f)

        params = cfg["params"]
        names = params["names"]
        b_low = np.array(params["lower_bounds"])
        b_up = np.array(params["upper_bounds"])
        steps = np.array(params["steps"])

        domain = InputDomain(dim=len(names), b_low=b_low, b_up=b_up, steps=steps)
        return domain, names

    def transform(self, X):
        """Transform a batch of points from real units to normalized [0, 1]^d units.

        Applies transform_feature to each dimension, which handles both continuous
        and discrete dimensions appropriately. For discrete dimensions, values are
        assumed to already align with the discrete grid in real units.

        Parameters:
            X (np.ndarray): Points in real units, shape (n_points, dim).

        Returns:
            np.ndarray: Points in normalized [0, 1]^d units, shape (n_points, dim).
        """
        X_scaled = X.copy()
        # Transform to [0,1]^d
        for n in range(self.dim):
            X_scaled[:, n] = self.transform_feature(n, X[:, n])
        return X_scaled

    def inverse_transform(self, X):
        """Transform a batch of points from normalized [0, 1]^d units to real units.

        Applies inverse_transform_feature to each dimension, which handles both continuous
        and discrete dimensions appropriately. For discrete dimensions, values are
        snapped to the nearest discrete level after denormalization.

        Parameters:
            X (np.ndarray): Points in normalized [0, 1]^d units, shape (n_points, dim).

        Returns:
            np.ndarray: Points in real units, shape (n_points, dim).
        """
        X_scaled = X.copy()
        # Scale back to parameter space
        for n in range(self.dim):
            X_scaled[:, n] = self.inverse_transform_feature(n, X[:, n])
        return X_scaled

    def transform_feature(self, feature_index, value):
        """Transform feature values from real units to normalized [0, 1] units.

        Floating-point errors near 0.0 and 1.0 are corrected to ensure proper boundary behavior.

        Parameters:
            feature_index (int): Index of the feature dimension to transform.
            value (float or np.ndarray): Feature value(s) in real units. Can be a scalar
                or array; output shape matches input shape.

        Returns:
            float or np.ndarray: Normalized value(s) in [0, 1]. Shape matches input value.
        """
        scalar_input = np.ndim(value) == 0
        # Scale to [0,1]
        value = (value - self.b_low[feature_index]) / (self.b_up[feature_index] - self.b_low[feature_index])
        # Catch floating point errors
        value = np.where(np.isclose(value, 0.0), 0.0, value)
        value = np.where(np.isclose(value, 1.0), 1.0, value)
        return value if not scalar_input else value.item()

    def inverse_transform_feature(self, feature_index, value):
        """Transform feature values from normalized [0, 1] units to real units.

        For discrete dimensions, values are snapped to the nearest discrete level in
        real units after denormalization, ensuring values align with the discrete grid.

        Parameters:
            feature_index (int): Index of the feature dimension to transform.
            value (float or np.ndarray): Normalized feature value(s) in [0, 1]. Can be a scalar
                or array; output shape matches input shape.

        Returns:
            float or np.ndarray: Feature value(s) in real units. Shape matches input value.
        """
        value = (self.b_up[feature_index] - self.b_low[feature_index]) * value + self.b_low[feature_index]
        # For discrete spaces, find nearest point.
        if feature_index in self.discrete_indices:
            value = np.rint(value / self.steps[feature_index]) * self.steps[feature_index]
        return value


@dataclass
class FidelityDomain:
    """
    Defines the fidelity space domain which is discrete

    One can assign costs to each fidelity here
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
        self.fidelity_weights = {input_dim: 1.0}
        self.target_fidelities = {input_dim: self.target_fidelity}
        self.fidelity_features = [{input_dim: f} for f in self.fidelities]

    def get_bounds(self):
        bounds = np.array([self.minimal_fidelity, self.target_fidelity])
        return bounds

    def project(self, XSs):
        return project_to_target_fidelity(X=XSs, target_fidelities=self.target_fidelities)
