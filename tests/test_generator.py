import numpy as np
import pytest
from millefeuille.domain import InputDomain
from millefeuille.generators import MetropolisHastingsGenerator
from millefeuille.simulator import PythonSimulator
from millefeuille.state import State
from millefeuille.surrogate import SingleFidelityGPSurrogate
from millefeuille.utils import run_generator_loop
from scipy.stats import qmc


@pytest.mark.unit
def test_MetropolisHastingsGenerator():
    """Test that the MetropolisHastingsGenerator produces samples that are
    distributed according to objective function."""

    class _TestSimulator(PythonSimulator):
        def __call__(self, indices, Xs, Ss=None):
            Ys = -0.5 * np.sum(Xs**2, axis=-1, keepdims=True)
            return None, Ys

    # Create a surrogate with a known kernel and hyperparameters
    surrogate = SingleFidelityGPSurrogate()

    # Create domain
    domain = InputDomain(
        dim=2,
        b_low=np.array([-2.0, -2.0]),
        b_up=np.array([2.0, 2.0]),
        steps=np.array([0.0, 0.0]),
    )

    # Create the simulator
    simulator = _TestSimulator()

    # Create a state with some random data
    Ninitial = 100
    Xs = qmc.Sobol(domain.dim, scramble=True, seed=12).random(Ninitial) * (domain.b_up - domain.b_low) + domain.b_low
    Ys = simulator(None, Xs)[1]
    state = State(input_domain=domain, index=np.arange(Ninitial), Xs=Xs, Ys=Ys)

    # Fit the surrogate to the initial data
    surrogate.fit(state)

    # Create the generator
    generator = MetropolisHastingsGenerator(
        domain=domain,
        surrogate=surrogate,
        proposal_std=0.05,
        n_burnin=10,
        n_steps=10,
        n_chains=100,
    )

    # Generate new candidates
    Nsamples = 20
    batch_size = 10
    new_state = run_generator_loop(
        Nsamples=Nsamples,
        batch_size=batch_size,
        simulator=simulator,
        generate_candidates=generator,
        state=state,
        verbose=False,
    )

    # Check that the new candidates are distributed according to the objective function
    # The distribution should approach a 2D Gaussian centered at (0, 0) with std ~1.0
    new_Xs = new_state.Xs[Ninitial:]

    assert new_Xs.shape == (Nsamples * batch_size, 2)
    assert np.all(np.isclose(np.mean(new_Xs, axis=0), np.array([0.0, 0.0]), atol=0.20))
    assert np.all(np.isclose(np.std(new_Xs, axis=0), np.array([1.0, 1.0]), atol=0.20))

    # Check that the new Xs are uncorrelated between dimensions
    corr = np.corrcoef(new_Xs, rowvar=False)
    assert np.isclose(corr[0, 1], 0.0, atol=0.20)
