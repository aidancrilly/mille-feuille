"""
Defines some useful utility functions which do not fit into the defined classes
"""

from .surrogate import *
from .acquisition import *

from botorch.optim import optimize_acqf_mixed

import time

DEFAULT_NUM_RESTARTS = 1
DEFAULT_RAW_SAMPLES  = 256

def generate_singlefidelity_batch(
    state,
    surrogate_model,
    acq_function,
    batch_size,
    num_restarts,
    raw_samples,
):
    if(acq_function == 'qLogExpectedImprovement'):
        # Expected Improvement
        ei = qLogExpectedImprovement(surrogate_model, state.best_value_transformed)
        X_next, _ = optimize_acqf(
            ei,
            bounds=state.get_bounds(),
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )
    else:
        print(f'Did not recognise acq_function in generate_singlefidelity_batch: {acq_function}')
        from sys import exit
        exit()

    return X_next

def generate_multifidelity_batch(
    state,
    surrogate_model,
    cost_model,
    generate_acq_function,
    batch_size,
    num_restarts,
    raw_samples,
    num_fantasies
):

    # Generate multi-fidelity acquisition function
    mfkg_acqf = generate_acq_function(
        state,
        surrogate_model,
        cost_model,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        num_fantasies=num_fantasies
        )
    
    # generate new candidates
    start = time.time()
    X_next, _ = optimize_acqf_mixed(
        acq_function=mfkg_acqf,
        bounds=state.get_bounds(),
        fixed_features_list=state.fidelity_features,
        q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options={"batch_limit": 5, "maxiter": 200},
    )
    print(f'Generating candidates, elasped time: {time.time()-start}')

    return X_next

def suggest_next_locations(
    batch_size,
    state,
    surrogate,
    acq_function,
    cost_model=None,
    num_restarts=DEFAULT_NUM_RESTARTS,
    raw_samples=DEFAULT_RAW_SAMPLES,
    num_fantasies=DEFAULT_NUM_FANTASIES
):
    # Check inputs
    if(state.l_MultiFidelity and cost_model is None):
        print('Error in suggest_next_locations:')
        print('Please provide cost model and generator of acquisition function to enable multi-fidelity...')
        from sys import exit
        exit()

    # Train the model
    surrogate.fit(state)

    # Create a batch
    if(state.l_MultiFidelity):
        X_next = generate_multifidelity_batch(
            state=state,
            surrogate_model=surrogate.model,
            cost_model=cost_model,
            generate_acq_function=acq_function,
            batch_size=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            num_fantasies=num_fantasies,
        )
    else:
        X_next = generate_singlefidelity_batch(
            state=state,
            surrogate_model=surrogate.model,
            acq_function=acq_function,
            batch_size=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )

    # Get to CPU and remove any AD info...
    X_next = X_next.detach().cpu().numpy()

    if(state.l_MultiFidelity):
        # Separate inputs and fidelities
        X_next, S_next = X_next[:,:-1],X_next[:,-1]

        # Transform to real domain
        X_next = state.domain.inverse_transform(X_next)

        return X_next, S_next
    else:
        # Transform to real domain
        X_next = state.domain.inverse_transform(X_next)

        return X_next