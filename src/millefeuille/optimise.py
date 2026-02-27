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
):
    """Optimise an acquisition function and return a batch of candidate points.

    Dispatches to :func:`botorch.optim.optimize_acqf` for single-fidelity
    problems and to :func:`botorch.optim.optimize_acqf_mixed` for
    multi-fidelity problems (where the fidelity is a fixed discrete feature).

    Parameters:
        state: Current :class:`~millefeuille.state.State`.
        acq_function: A BoTorch acquisition function instance.
        batch_size: Number of candidates to generate (``q``).
        num_restarts: Number of random restarts for the optimiser.
        raw_samples: Number of raw samples used to seed the optimiser.
        optimizer_options: Dict of options forwarded to the BoTorch optimiser
            (e.g. ``{"maxiter": 200}``).  May be ``None``.

    Returns:
        torch.Tensor: Candidate tensor of shape ``(batch_size, d)`` on the
        unit hypercube (with fidelity column appended for multi-fidelity).
    """
    if state.l_MultiFidelity:
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
        X_next, _ = optimize_acqf(
            acq_function,
            bounds=state.get_bounds(),
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options=optimizer_options,
        )

    return X_next


def suggest_next_locations(
    batch_size,
    state,
    acq_function,
    num_restarts=DEFAULT_NUM_RESTARTS,
    raw_samples=DEFAULT_RAW_SAMPLES,
    optimizer_options=None,
    verbose=False,
):
    """Suggest the next batch of input locations to evaluate.

    Wraps :func:`generate_batch` to optimise the acquisition function and
    transform the resulting candidates back to the original parameter space.

    Parameters:
        batch_size: Number of new locations to suggest.
        state: Current :class:`~millefeuille.state.State`.
        acq_function: A BoTorch acquisition function instance.
        num_restarts: Number of random restarts for the inner optimiser
            (default: ``10``).
        raw_samples: Number of raw samples used to seed the optimiser
            (default: ``256``).
        optimizer_options: Optional dict passed to the BoTorch optimiser.
        verbose: If ``True``, prints timing information.

    Returns:
        For **single-fidelity** problems:
            np.ndarray: ``X_next`` of shape ``(batch_size, dim)`` in parameter
            space.

        For **multi-fidelity** problems:
            tuple[np.ndarray, np.ndarray]: ``(X_next, S_next)`` where
            ``X_next`` has shape ``(batch_size, dim)`` and ``S_next`` has
            shape ``(batch_size, 1)`` containing the recommended fidelity
            indices.
    """
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
