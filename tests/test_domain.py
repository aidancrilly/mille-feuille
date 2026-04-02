import numpy as np
import pytest
import pytest_cases
from millefeuille import InputDomain
from millefeuille.domain import ScaleFactorInputDomain


@pytest_cases.fixture(params=[100, 1000])
def nsample(request):
    return request.param


@pytest_cases.fixture(params=[2, 5])
def dim(request):
    return request.param


@pytest_cases.fixture()
def b_low(dim):
    return np.random.rand(dim)


@pytest_cases.fixture()
def b_up(dim, b_low):
    return b_low + np.random.rand(dim) + 1e-2


@pytest.mark.unit
def test_continuous_domain(nsample, dim, b_low, b_up):
    domain = InputDomain(dim=dim, b_low=b_low, b_up=b_up, steps=np.zeros_like(b_low))

    X = b_low[None, :] + (b_up[None, :] - b_low[None, :]) * np.random.rand(nsample, dim)
    X_unit = domain.transform(X)

    assert (X_unit >= 0.0).all()

    assert (X_unit <= 1.0).all()

    X_back = domain.inverse_transform(X_unit)

    assert np.isclose(X_back, X).all()


@pytest.mark.unit
def test_mixed_domain(nsample, dim, b_low, b_up):
    steps = (b_up - b_low) / 100
    steps[: dim // 2] = 0.0

    domain = InputDomain(dim=dim, b_low=b_low, b_up=b_up, steps=steps)

    X = b_low[None, :] + (b_up[None, :] - b_low[None, :]) * np.random.rand(nsample, dim)
    X_unit = domain.transform(X)

    assert (X_unit >= 0.0).all()

    assert (X_unit <= 1.0).all()

    X_back = domain.inverse_transform(X_unit)

    X_continuous = X_back[:, : dim // 2]
    X_discrete = X_back[:, dim // 2 :]

    assert np.isclose(X_continuous, X[:, : dim // 2]).all()

    for i in range(X_discrete.shape[1]):
        assert np.isclose(X_discrete[:, i], X[:, dim // 2 + i], rtol=0.0, atol=steps[dim // 2 + i] / 2.0).all()


@pytest.mark.unit
def test_transform_feature(dim, b_low, b_up):
    domain = InputDomain(dim=dim, b_low=b_low, b_up=b_up, steps=np.zeros_like(b_low))

    for n in range(dim):
        # Real value at midpoint should map to 0.5
        mid_real = 0.5 * (b_low[n] + b_up[n])
        mid_unit = domain.transform_feature(n, mid_real)
        assert np.isclose(mid_unit, 0.5), f"transform_feature midpoint failed for dim {n}"

        # Bounds
        assert np.isclose(domain.transform_feature(n, b_low[n]), 0.0)
        assert np.isclose(domain.transform_feature(n, b_up[n]), 1.0)

        # Round-trip
        for unit_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
            real_val = domain.inverse_transform_feature(n, unit_val)
            assert np.isclose(domain.transform_feature(n, real_val), unit_val)


@pytest.mark.unit
def test_scale_factor_domain_continuous(nsample, dim, b_low, b_up):
    """Scale factor applied to continuous dims: check output range and scaling."""
    domain = ScaleFactorInputDomain(dim=dim, b_low=b_low, b_up=b_up, steps=np.zeros_like(b_low))

    X_unit = np.random.rand(nsample, dim)
    X_real = domain.inverse_transform(X_unit)

    # Dimension 0 is the scale factor; recover it
    scale_factor_real = (b_up[0] - b_low[0]) * X_unit[:, 0] + b_low[0]
    assert np.allclose(X_real[:, 0], scale_factor_real)

    # Dimensions 1..N should equal (physical value) * scale_factor
    for n in range(1, dim):
        expected = ((b_up[n] - b_low[n]) * X_unit[:, n] + b_low[n]) * scale_factor_real
        assert np.allclose(X_real[:, n], expected)


@pytest.mark.unit
def test_scale_factor_domain_discrete(nsample, dim, b_low, b_up):
    """Scale factor applied before snapping: discrete dims should lie on their grids."""
    steps = np.zeros_like(b_low)
    # Make all dimensions except dim 0 discrete
    for i in range(1, dim):
        steps[i] = (b_up[i] - b_low[i]) / 10.0

    domain = ScaleFactorInputDomain(dim=dim, b_low=b_low, b_up=b_up, steps=steps)

    X_unit = np.random.rand(nsample, dim)
    X_real = domain.inverse_transform(X_unit)

    scale_factor_real = (b_up[0] - b_low[0]) * X_unit[:, 0] + b_low[0]
    assert np.allclose(X_real[:, 0], scale_factor_real)

    # Discrete dimensions must be multiples of their step size
    for n in domain.discrete_indices:
        assert np.allclose(X_real[:, n] / steps[n], np.rint(X_real[:, n] / steps[n])), (
            f"Discrete dim {n} values not on grid"
        )


@pytest.mark.unit
def test_scale_factor_domain_inherits_transform(nsample, dim, b_low, b_up):
    """transform() should behave identically to InputDomain.transform()."""
    domain_sf = ScaleFactorInputDomain(dim=dim, b_low=b_low, b_up=b_up, steps=np.zeros_like(b_low))
    domain_base = InputDomain(dim=dim, b_low=b_low, b_up=b_up, steps=np.zeros_like(b_low))

    X = b_low[None, :] + (b_up[None, :] - b_low[None, :]) * np.random.rand(nsample, dim)
    assert np.allclose(domain_sf.transform(X), domain_base.transform(X))
