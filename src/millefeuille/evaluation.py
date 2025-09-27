import timeit

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from .state import State
from .surrogate import (
    DEFAULT_LENGTHSCALE_INTERVAL,
    DEFAULT_NOISE_INTERVAL,
    MultiFidelityGPSurrogate,
    SingleFidelityGPSurrogate,
)

# ------------------------------
# Metrics computation
# ------------------------------


def compute_metrics(y_true, y_pred):
    """Compute R2, MAE, RMSE for given predictions."""
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
    }


def evaluate_GP_scores_multi_fid(surrogate, state, Xs_test, Ys_test, Ss_test, verbose=True):
    """Compute metrics for low and high fidelity."""
    outputs = {}
    results = []
    for label, mask in zip(["low", "high"], [0, 1], strict=False):
        X, y_true = Xs_test[Ss_test == mask], Ys_test[Ss_test == mask]
        pred_out = surrogate.predict(state, X)
        y_pred = np.asarray(pred_out[mask]["mean"]).reshape(-1)
        y_std = np.asarray(pred_out[mask]["std"]).reshape(-1)
        outputs[label] = compute_metrics(y_true, y_pred)
        results.extend([y_true, y_pred, y_std])
        if verbose:
            print(
                f"Fidelity {label}: R2={outputs[label]['R2']:.3f}, "
                f"MAE={outputs[label]['MAE']:.3f}, RMSE={outputs[label]['RMSE']:.3f}"
            )
    return outputs, tuple(results)


def evaluate_GP_scores_single_fid(surrogate, state, Xs_test, Ys_test, verbose=True):
    """Compute metrics for single-fidelity GP."""
    pred_out = surrogate.predict(state, Xs_test)
    y_pred = np.asarray(pred_out["mean"]).reshape(-1)
    y_std = np.asarray(pred_out["std"]).reshape(-1)
    metrics = compute_metrics(Ys_test, y_pred)
    if verbose:
        print(f"R2={metrics['R2']:.3f}, MAE={metrics['MAE']:.3f}, RMSE={metrics['RMSE']:.3f}")
    return metrics, (Ys_test, y_pred, y_std)


# ------------------------------
# Plotting
# ------------------------------


def plot_predictions(ax_pred, ax_resid, metrics_dict, title_suffix=""):
    """
    General plotting function.
    metrics_dict: dictionary of {'label': (y_true, y_pred, y_std, metrics)}
    """
    for label, (y_true, y_pred, y_std, metrics) in metrics_dict.items():
        y_true, y_pred, y_std = map(np.squeeze, (y_true, y_pred, y_std))
        color = "blue" if label == "low" else "red"
        # Predictions plot
        ax_pred.errorbar(
            y_true, y_pred, yerr=y_std, fmt="o", alpha=0.3, c=color, label=f"{label} fidelity (R²={metrics['R2']:.3f})"
        )
        # Residuals plot
        resid = y_true - y_pred
        ax_resid.hist(resid, bins=30, alpha=0.5, color=color, label=f"{label} fidelity (MAE={metrics['MAE']:.3f})")

    ymin = min(np.min(v[0]) for v in metrics_dict.values())
    ymax = max(np.max(v[0]) for v in metrics_dict.values())
    ax_pred.plot([ymin, ymax], [ymin, ymax], "k--")
    ax_pred.set_xlabel("Truth")
    ax_pred.set_ylabel("Prediction")
    ax_pred.set_title(f"Truth vs Prediction {title_suffix}")
    ax_pred.legend()
    ax_pred.grid()

    ax_resid.set_xlabel("Residual")
    ax_resid.set_ylabel("Frequency")
    ax_resid.set_title(f"Residuals {title_suffix}")
    ax_resid.legend()
    ax_resid.grid()


# ------------------------------
# k-fold evaluation
# ------------------------------


def kfold_GP_evaluation(
    state,
    k_folds=5,
    lengthscale_interval=DEFAULT_LENGTHSCALE_INTERVAL,
    noise_interval=DEFAULT_NOISE_INTERVAL,
    fname=None,
    **kernel_kwargs,
):
    """k-fold CV for GP surrogates."""
    Xs, Ys, Is = state.Xs, state.Ys, state.index
    is_multi = state.l_MultiFidelity
    Ss = state.Ss[:, 0] if is_multi else None

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    cv_metrics = []

    fig, axes = plt.subplots(k_folds, 2, figsize=(10, 3.5 * k_folds))
    axes = axes if k_folds > 1 else np.array([axes])

    for i, (train_idx, test_idx) in enumerate(kf.split(Xs)):
        X_train, X_test = Xs[train_idx], Xs[test_idx]
        Y_train, Y_test = Ys[train_idx], Ys[test_idx]
        I_train = Is[train_idx]

        # Prepare surrogate
        state_new = State(
            state.input_domain,
            index=I_train,
            Xs=X_train,
            Ys=Y_train,
            Ss=Ss[train_idx] if is_multi else None,
            fidelity_domain=state.fidelity_domain,
        )
        SurrogateClass = MultiFidelityGPSurrogate if is_multi else SingleFidelityGPSurrogate
        surrogate = SurrogateClass(
            lengthscale_interval=lengthscale_interval, noise_interval=noise_interval, **kernel_kwargs
        )

        elapsed = timeit.timeit(lambda _, surrogate=surrogate, state=state_new: surrogate.fit(state), number=1)
        print(f"Fold {i + 1}: fit time = {elapsed:.2f}s")

        # Evaluate
        if is_multi:
            metrics, results = evaluate_GP_scores_multi_fid(surrogate, state_new, X_test, Y_test, Ss[test_idx])
            plot_data = {"low": (*results[:3], metrics["low"]), "high": (*results[3:], metrics["high"])}
        else:
            metrics, results = evaluate_GP_scores_single_fid(surrogate, state_new, X_test, Y_test)
            plot_data = {"single": (*results, metrics)}

        ax_pred, ax_resid = axes[i] if k_folds > 1 else axes
        plot_predictions(ax_pred, ax_resid, plot_data, title_suffix=f"(Fold {i + 1})")
        cv_metrics.append(metrics)

    plt.tight_layout()
    plt.show()

    if fname:
        fig.savefig(fname + "_kfold.png", dpi=200)

    # ------------------------------
    # Summary of CV R2
    # ------------------------------
    if is_multi:
        mean_r2_low = np.mean([m["low"]["R2"] for m in cv_metrics])
        std_r2_low = np.std([m["low"]["R2"] for m in cv_metrics])
        mean_r2_high = np.mean([m["high"]["R2"] for m in cv_metrics])
        std_r2_high = np.std([m["high"]["R2"] for m in cv_metrics])
        print(f"Average CV R2 low fidelity: {mean_r2_low:.3f} ± {std_r2_low:.3f}")
        print(f"Average CV R2 high fidelity: {mean_r2_high:.3f} ± {std_r2_high:.3f}")
    else:
        mean_r2 = np.mean([m["R2"] for m in cv_metrics])
        std_r2 = np.std([m["R2"] for m in cv_metrics])
        print(f"Average CV R2: {mean_r2:.3f} ± {std_r2:.3f}")

    return cv_metrics


# ------------------------------
# Full surrogate evaluation
# ------------------------------


def evaluate_GP_surrogate(
    state,
    k_folds=5,
    lengthscale_interval=DEFAULT_LENGTHSCALE_INTERVAL,
    noise_interval=DEFAULT_NOISE_INTERVAL,
    fname=None,
    **kernel_kwargs,
):
    """Master evaluation function."""
    is_multi = state.l_MultiFidelity
    SurrogateClass = MultiFidelityGPSurrogate if is_multi else SingleFidelityGPSurrogate
    surrogate = SurrogateClass(
        lengthscale_interval=lengthscale_interval, noise_interval=noise_interval, **kernel_kwargs
    )

    elapsed = timeit.timeit(lambda: surrogate.fit(state), number=1)
    print(f"Fitted full surrogate in {elapsed:.3f}s")

    # Evaluate full data
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    if is_multi:
        metrics, results = evaluate_GP_scores_multi_fid(surrogate, state, state.Xs, state.Ys, state.Ss[:, 0])
        plot_data = {"low": (*results[:3], metrics["low"]), "high": (*results[3:], metrics["high"])}
    else:
        metrics, results = evaluate_GP_scores_single_fid(surrogate, state, state.Xs, state.Ys)
        plot_data = {"single": (*results, metrics)}

    plot_predictions(axes[0], axes[1], plot_data, title_suffix="(Full dataset)")
    plt.tight_layout()
    plt.show()

    if fname:
        fig.savefig(fname + "_all.png", dpi=200)

    # Run k-fold CV
    cv_metrics = kfold_GP_evaluation(
        state,
        k_folds=k_folds,
        lengthscale_interval=lengthscale_interval,
        noise_interval=noise_interval,
        fname=fname,
        **kernel_kwargs,
    )
    return metrics, cv_metrics
