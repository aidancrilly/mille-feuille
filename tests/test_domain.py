import numpy as np
import pytest
import pytest_cases
from millefeuille import InputDomain


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
