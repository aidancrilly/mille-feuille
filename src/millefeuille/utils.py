import numpy as np
from scipy.stats import norm

from .optimise import *
from .simulator import *
from .surrogate import BaseSurrogate

"""
Defines some useful utility functions which do not fit into the defined classes
"""


def probabilistic_threshold_sampling(
    domain,
    state,
    sampler,
    surrogate,
    initial_samples,
    threshold_value,
    target_fidelity=None,
    target_key=None,
    random_draws=None,
):
    """
    Samples points using a GP surrogate and filters them probabilistically
    based on the predicted probability of exceeding a threshold.

    Parameters:
        domain: InputDomain object
        state: State object
        sampler: QMC sampler (e.g. Sobol)
        surrogate: mille-feuille-compatible surrogate (must support predict(domain, X))
        initial_samples: number of samples to draw
        threshold_value: threshold value to compare against
        target_fidelity: int — selects surrogate fidelity
        target_key: int or str — selects surrogate output
        random_draws: optional np.ndarray of uniform(0,1) values of shape (initial_samples,)
                      if None, will be generated internally

    Returns:
        x_all: all sampled input points (shape: NxD)
        y_pred: np.ndarray of predicted means (selected column)
        prob: predicted P(y > threshold) for each point
        mask: boolean array of points that passed the stochastic filter
    """
    # Generate inputs
    x_unit = sampler.random(initial_samples)
    x_all = domain.inverse_transform(x_unit)

    # Predict mean and std from GP
    predictions = surrogate.predict(state, x_all)

    if target_key is not None:
        prediction = predictions[target_key]
    else:
        if target_fidelity is not None:
            prediction = predictions[target_fidelity]
        else:
            if "mean" not in predictions.keys():
                raise ValueError(
                    "mean missing from predictions.keys(), did you miss target_fidelity or target_key inputs?"
                )
            prediction = predictions

    mean, std = prediction["mean"], prediction["std"]

    # Compute probability P(y > threshold)
    mean = mean.flatten()
    std = std.flatten()
    prob = 1.0 - norm.cdf(threshold_value, loc=mean, scale=std)

    # Generate random draws
    if random_draws is None:
        random_draws = np.random.rand(initial_samples)

    mask = prob > random_draws
    y_pred = mean

    return x_all, y_pred, prob, mask


def surrogate_threshold_sampling(
    domain, state, sampler, surrogate, initial_samples, surrogate_threshold, target_fidelity=None, target_key=None
):
    """
    Draws samples from the sampler, evaluates the surrogate model,
    and selects those above a specified threshold.

    Parameters:
        domain: InputDomain object
        state: State object
        sampler: QMC sampler (e.g., Sobol)
        surrogate: mille-feuille-style surrogate with .predict(domain, Xs) method
        initial_samples: int, number of candidate points
        surrogate_threshold: float, threshold to apply on predicted mean
        target_fidelity: int — selects surrogate fidelity
        target_key: int or str — selects surrogate output

    Returns:
        x_all: np.ndarray of all candidate input samples
        y_pred: np.ndarray of predicted means (selected column)
        mask: np.ndarray of boolean values where prediction > threshold
    """
    # Draw samples in unit hypercube, transform to input space
    x_unit = sampler.random(initial_samples)
    x_all = domain.inverse_transform(x_unit)

    # Predict mean and std from GP
    predictions = surrogate.predict(state, x_all)

    if target_key is not None:
        prediction = predictions[target_key]
    else:
        if target_fidelity is not None:
            prediction = predictions[target_fidelity]
        else:
            if "mean" not in predictions.keys():
                raise ValueError(
                    "mean missing from predictions.keys(), did you miss target_fidelity or target_key inputs?"
                )
            prediction = predictions

    mean = prediction["mean"]

    y_pred = mean
    mask = mean > surrogate_threshold

    return x_all, y_pred, mask


def run_Bayesian_optimiser(
    Nsamples,
    batch_size,
    generate_acq_function,
    state,
    surrogate,
    simulator,
    scheduler=None,
    csv_name=None,
    verbose=False,
    **kwargs,
):
    """Run a Bayesian optimisation loop for a fixed number of iterations.

    At each iteration the surrogate is re-fitted to all current samples, a
    new batch of candidate points is suggested by optimising the acquisition
    function, the simulator is called to evaluate those points, and the
    state is updated with the new observations.

    Parameters:
        Nsamples: Number of optimisation iterations (surrogate fit + suggest +
            evaluate cycles).
        batch_size: Number of new points to evaluate per iteration.
        generate_acq_function: Callable with signature
            ``(surrogate, state) -> acq_function`` that returns a BoTorch
            acquisition function given the current surrogate and state.
        state: :class:`~millefeuille.state.State` holding the initial data.
            Modified **in-place** and also returned.
        surrogate: A :class:`~millefeuille.surrogate.BaseSurrogate` instance.
        simulator: A :class:`~millefeuille.simulator.PythonSimulator` or
            :class:`~millefeuille.simulator.ExectuableSimulator` instance.
        scheduler: Scheduler required when *simulator* is an
            :class:`~millefeuille.simulator.ExectuableSimulator`; ``None``
            otherwise.
        csv_name: If provided, the state is incrementally saved to this CSV
            file after each iteration via :meth:`~millefeuille.state.State.to_csv`.
        verbose: If ``True``, passes verbose flag to
            :func:`~millefeuille.optimise.suggest_next_locations`.
        **kwargs: Additional keyword arguments forwarded to
            :func:`~millefeuille.optimise.suggest_next_locations`.

    Returns:
        :class:`~millefeuille.state.State`: The updated state containing all
        samples collected during the loop.

    Raises:
        Exception: If *simulator* is an
            :class:`~millefeuille.simulator.ExectuableSimulator` but no
            *scheduler* is provided.

    Example:
        >>> state = run_Bayesian_optimiser(
        ...     Nsamples=10,
        ...     batch_size=2,
        ...     generate_acq_function=my_acq_fn_factory,
        ...     state=initial_state,
        ...     surrogate=gp_surrogate,
        ...     simulator=my_simulator,
        ... )
    """
    if isinstance(simulator, ExectuableSimulator) and scheduler is None:
        print("If simulator is an ExecutableSimulator, you must provide a scheduler")
        raise Exception
    assert isinstance(surrogate, BaseSurrogate)

    for _ in range(Nsamples):
        surrogate.fit(state)
        acq_function = generate_acq_function(surrogate, state)

        if state.l_MultiFidelity:
            X_next, S_next = suggest_next_locations(
                batch_size, state, acq_function=acq_function, verbose=verbose, **kwargs
            )
        else:
            X_next = suggest_next_locations(batch_size, state, acq_function=acq_function, verbose=verbose, **kwargs)
            S_next = None

        index_next = state.index[-1] + np.arange(batch_size) + 1
        if isinstance(simulator, ExectuableSimulator):
            P_next, Y_next = simulator(index_next, X_next, scheduler, Ss=S_next)
        elif isinstance(simulator, PythonSimulator):
            P_next, Y_next = simulator(index_next, X_next, Ss=S_next)
        else:
            print("simulator class not recognised, inherit for mille-feuille Simulator classes...")

        state.update(index_next, X_next=X_next, Y_next=Y_next, P_next=P_next, S_next=S_next)
        if csv_name is not None:
            state.to_csv(csv_name)

    return state
