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
def test_scale_factor_inverse_transform(nsample, dim, b_low, b_up):
    """inverse_transform maps unit hypercube to physical space with scale factor coupling."""
    domain = ScaleFactorInputDomain(dim=dim, b_low=b_low, b_up=b_up, steps=np.zeros_like(b_low))

    X_unit = np.random.rand(nsample, dim)
    X_real = domain.inverse_transform(X_unit)

    # Dimension 0 is the scale factor — should be a simple linear map
    scale_factor = (b_up[0] - b_low[0]) * X_unit[:, 0] + b_low[0]
    assert np.allclose(X_real[:, 0], scale_factor)

    # Dimensions 1..N should be (unscaled physical value) * scale_factor
    for n in range(1, dim):
        unscaled = (b_up[n] - b_low[n]) * X_unit[:, n] + b_low[n]
        assert np.allclose(X_real[:, n], unscaled * scale_factor)


@pytest.mark.unit
def test_scale_factor_roundtrip(nsample, dim, b_low, b_up):
    """transform(inverse_transform(X)) should recover the original unit-cube points."""
    domain = ScaleFactorInputDomain(dim=dim, b_low=b_low, b_up=b_up, steps=np.zeros_like(b_low))

    X_unit = np.random.rand(nsample, dim)
    X_real = domain.inverse_transform(X_unit)
    X_recovered = domain.transform(X_real)

    assert np.allclose(X_recovered, X_unit)


@pytest.mark.unit
def test_scale_factor_transform_undoes_coupling(nsample, dim, b_low, b_up):
    """transform should divide dims 1..N by the scale factor before normalising."""
    domain = ScaleFactorInputDomain(dim=dim, b_low=b_low, b_up=b_up, steps=np.zeros_like(b_low))
    base_domain = InputDomain(dim=dim, b_low=b_low, b_up=b_up, steps=np.zeros_like(b_low))

    # Build a physical-space array where dims 1..N include the scale factor
    X_unit = np.random.rand(nsample, dim)
    X_real = domain.inverse_transform(X_unit)

    # After ScaleFactorInputDomain.transform undoes the coupling, the result
    # should differ from the base InputDomain.transform (unless scale factor == 1)
    X_sf = domain.transform(X_real)
    X_base = base_domain.transform(X_real)

    # Dim 0 should be identical in both
    assert np.allclose(X_sf[:, 0], X_base[:, 0])

    # Dims 1..N should generally differ (scale factor is not 1 for random bounds)
    # but the scale-factor version should recover the original unit-cube values
    assert np.allclose(X_sf, X_unit)


@pytest.mark.unit
def test_scale_factor_feature_roundtrip(dim, b_low, b_up):
    """Per-feature transform/inverse_transform with explicit scale_factor argument."""
    domain = ScaleFactorInputDomain(dim=dim, b_low=b_low, b_up=b_up, steps=np.zeros_like(b_low))

    sf_physical = 0.5 * (b_low[0] + b_up[0])  # pick a concrete scale factor value

    for n in range(dim):
        for unit_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
            real_val = domain.inverse_transform_feature(n, unit_val, scale_factor=sf_physical if n > 0 else None)
            recovered = domain.transform_feature(n, real_val, scale_factor=sf_physical if n > 0 else None)
            assert np.isclose(recovered, unit_val), f"Round-trip failed for dim {n}, unit_val {unit_val}"


@pytest.mark.unit
def test_scale_factor_discrete_snapping(nsample, dim, b_low, b_up):
    """Discrete dims should snap to their grids after the scale factor is applied."""
    steps = np.zeros_like(b_low)
    for i in range(1, dim):
        steps[i] = (b_up[i] - b_low[i]) / 10.0

    domain = ScaleFactorInputDomain(dim=dim, b_low=b_low, b_up=b_up, steps=steps)

    X_unit = np.random.rand(nsample, dim)
    X_real = domain.inverse_transform(X_unit)

    # Discrete dimensions must be multiples of their step size
    for n in domain.discrete_indices:
        remainder = np.abs(X_real[:, n] / steps[n] - np.rint(X_real[:, n] / steps[n]))
        assert np.allclose(remainder, 0.0), f"Discrete dim {n} values not on grid"
