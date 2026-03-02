import time

from botorch.optim import optimize_acqf, optimize_acqf_mixed

from .state import State

DEFAULT_NUM_RESTARTS = 10
DEFAULT_RAW_SAMPLES = 256


def generate_batch(
    state,
    acq_function,
    batch_size,
    num_restarts,
    raw_samples,
    optimizer_options,
    fixed_features=None,
):
    # Generate new candidates
    if state.l_MultiFidelity:
        if fixed_features is not None:
            raise ValueError("fixed_features is only supported for single fidelity optimisation")
        X_next, _ = optimize_acqf_mixed(
            acq_function=acq_function,
            bounds=state.get_bounds(),
            fixed_features_list=[state.fidelity_domain.target_fidelities],
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options=optimizer_options,
        )
    else:
        # Transform fixed_features values from real units to normalised [0, 1] units
        normalised_fixed_features = None
        if fixed_features is not None:
            normalised_fixed_features = {
                idx: state.input_domain.transform_feature(idx, val) for idx, val in fixed_features.items()
            }
        X_next, _ = optimize_acqf(
            acq_function,
            bounds=state.get_bounds(),
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options=optimizer_options,
            fixed_features=normalised_fixed_features,
        )

    return X_next


def suggest_next_locations(
    batch_size,
    state,
    acq_function,
    num_restarts=DEFAULT_NUM_RESTARTS,
    raw_samples=DEFAULT_RAW_SAMPLES,
    optimizer_options=None,
    fixed_features=None,
    verbose=False,
):
    # Check inputs
    assert isinstance(state, State)

    # Create a batch
    if verbose:
        start = time.time()
        print("Generating candidates ...")

    X_next = generate_batch(
        state=state,
        acq_function=acq_function,
        batch_size=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        optimizer_options=optimizer_options,
        fixed_features=fixed_features,
    )

    if verbose:
        print(f"Candidates generated in {time.time() - start} s")

    # Get to CPU and remove any AD info...
    X_next = X_next.detach().cpu().numpy()

    if state.l_MultiFidelity:
        # Separate inputs and fidelities
        X_next, S_next = X_next[:, :-1], X_next[:, -1:]

        # Transform to real domain
        X_next = state.inverse_transform_X(X_next)

        return X_next, S_next
    else:
        # Transform to real domain
        X_next = state.inverse_transform_X(X_next)

        return X_next
