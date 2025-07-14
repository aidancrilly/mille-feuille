import numpy as np

"""
Code to assist in gathering initial data before beginning optimisation
"""


def generate_initial_sample(domain, sampler, initial_samples, batch_size=1):
    # Make sure divisible by Nbatch and greater than initial_samples
    initial_samples = int(np.ceil(initial_samples / batch_size)) * batch_size
    # Get sample
    X = sampler.random(initial_samples)
    if domain.dim == 1:
        X = X.reshape(-1, 1)
    X = domain.inverse_transform(X)
    return X, initial_samples
