"""
Defines some useful utility functions which do not fit into the defined classes
"""
import numpy as np

from .simulator import *
from .optimise import *

from scipy.stats import norm
import numpy as np

def probabilistic_threshold_sampling(
    domain,
    sampler,
    surrogate,
    initial_samples,
    threshold_value,
    target_key=0,
    random_draws=None
):
    """
    Samples points using a GP surrogate and filters them probabilistically
    based on the predicted probability of exceeding a threshold.

    Parameters:
        domain: InputDomain object
        sampler: QMC sampler (e.g. Sobol)
        surrogate: mille-feuille-compatible surrogate (must support predict(domain, X))
        initial_samples: number of samples to draw
        threshold_value: threshold value to compare against
        target_key: int or str — selects surrogate output (default = 0)
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
    prediction = surrogate.predict(domain, x_all)

    if isinstance(prediction, dict):
        if isinstance(target_key, int):
            key = list(prediction.keys())[target_key]
        else:
            key = target_key
        mean, std = prediction[key]
    else:
        mean, std = prediction[:, 0], prediction[:, 1]

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
        domain,
        sampler,
        surrogate,
        initial_samples,
        surrogate_threshold,
        target_key=0):
    """
    Draws samples from the sampler, evaluates the surrogate model,
    and selects those above a specified threshold.

    Parameters:
        domain: InputDomain object
        sampler: QMC sampler (e.g., Sobol)
        surrogate: mille-feuille-style surrogate with .predict(domain, Xs) method
        initial_samples: int, number of candidate points
        surrogate_threshold: float, threshold to apply on predicted mean
        target_key: int or str — selects surrogate output (default = 0)

    Returns:
        x_all: np.ndarray of all candidate input samples
        y_pred: np.ndarray of predicted means (selected column)
        mask: np.ndarray of boolean values where prediction > threshold
    """
    # Draw samples in unit hypercube, transform to input space
    x_unit = sampler.random(initial_samples)
    x_all = domain.inverse_transform(x_unit)

    # Predict mean and std from GP
    prediction = surrogate.predict(domain, x_all)

    if isinstance(prediction, dict):
        if isinstance(target_key, int):
            key = list(prediction.keys())[target_key]
        else:
            key = target_key
        mean, _ = prediction[key]
    else:
        mean, _ = prediction[:, 0], prediction[:, 1]

    y_pred = mean
    mask = mean > surrogate_threshold

    return x_all, y_pred, mask

def singlefidelity_serial_BO_run(Nsamples,acq_function,state,surrogate,simulator,scheduler=None,csv_name=None):
    if(isinstance(simulator,ExectuableSimulator) and scheduler is None):
        print('If simulator is an ExecutableSimulator, you must provide a scheduler')
        raise Exception

    batch_size = 1
    for _ in range(Nsamples):
        X_next = suggest_next_locations(batch_size,state,surrogate,
        acq_function=acq_function)

        index_next = np.array([state.index[-1]+1])
        if(isinstance(simulator,ExectuableSimulator)):
            P_next, Y_next = simulator(index_next, X_next, scheduler)
        elif(isinstance(simulator,PythonSimulator)):
            P_next, Y_next = simulator(index_next, X_next)
        else:
            print('simulator class not recognised, inherit for mille-feuille Simulator classes...')

        state.update(index_next,X_next=X_next,Y_next=Y_next,P_next=P_next)
        if(csv_name is not None):
            state.to_csv(csv_name)

    return state

def multifidelity_serial_BO_run(Nsamples,acq_function,cost_model,state,surrogate,simulator,scheduler=None,csv_name=None):
    if(isinstance(simulator,ExectuableSimulator) and scheduler is None):
        print('If simulator is an ExecutableSimulator, you must provide a scheduler')
        raise Exception

    batch_size = 1
    for _ in range(Nsamples):
        X_next,S_next = suggest_next_locations(batch_size,state,surrogate,
        acq_function=acq_function,
        cost_model=cost_model)

        index_next = np.array([state.index[-1]+1])
        if(isinstance(simulator,ExectuableSimulator)):
            P_next, Y_next = simulator(index_next, X_next, scheduler, Ss = S_next)
        elif(isinstance(simulator,PythonSimulator)):
            P_next, Y_next = simulator(index_next, X_next, Ss = S_next)
        else:
            print('simulator class not recognised, inherit for mille-feuille Simulator classes...')

        state.update(index_next,X_next=X_next,Y_next=Y_next,P_next=P_next,S_next=S_next)
        if(csv_name is not None):
            state.to_csv(csv_name)

    return state
