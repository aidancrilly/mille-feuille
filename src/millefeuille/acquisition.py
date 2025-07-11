from botorch.acquisition import PosteriorMean
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.optim import optimize_acqf

"""
Defines number of acquisition functions beyond the standards in BOtorch
"""

# Multi-Fidelity hyperparameters
DEFAULT_NUM_FANTASIES = 64


def generate_MFKG_acqf(state, surrogate_model, cost_model, num_restarts, raw_samples, num_fantasies):
    curr_val_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(surrogate_model),
        d=state.dim + 1,
        columns=[state.dim],
        values=[state.target_fidelity],
    )

    bounds = state.get_bounds()

    _, current_value = optimize_acqf(
        acq_function=curr_val_acqf,
        bounds=bounds[:, :-1],
        q=1,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options={"batch_limit": 10, "maxiter": 200},
    )

    mfkg = qMultiFidelityKnowledgeGradient(
        model=surrogate_model,
        num_fantasies=num_fantasies,
        current_value=current_value,
        cost_aware_utility=cost_model,
        project=state.fidelity_project,
    )

    return mfkg
