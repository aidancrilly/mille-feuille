"""
mille-feuille Dashboard
=======================

A Streamlit app for inspecting optimisation state, surrogate quality, and
optimal-design landscapes.

Can be launched via the installed console script::

    mf-dashboard                           # local
    mf-dashboard --server.port 8502        # custom port

Or directly with streamlit::

    streamlit run src/millefeuille/dashboard/app.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from botorch.acquisition import PosteriorMean
from botorch.optim import optimize_acqf
from sklearn.model_selection import KFold

from millefeuille.domain import InputDomain
from millefeuille.state import State
from millefeuille.surrogate import (
    SingleFidelityGPSurrogate,
    MultiFidelityGPSurrogate,
    SingleFidelityRandomForestSurrogate,
    SingleFidelityEnsembleSurrogate,
)
from millefeuille.evaluation import (
    evaluate_GP_scores_single_fid,
    evaluate_GP_scores_multi_fid,
    compute_metrics,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="mille-feuille Dashboard", layout="wide")
st.title("🍰 mille-feuille Dashboard")

# ---------------------------------------------------------------------------
# Sidebar — file / configuration selectors
# ---------------------------------------------------------------------------
st.sidebar.header("Configuration")

domain_file = st.sidebar.file_uploader("Domain JSON", type=["json"])
state_file = st.sidebar.file_uploader("State file (.h5 / .hdf5 / .csv)", type=["h5", "hdf5", "csv"])

SURROGATE_OPTIONS = {
    "SingleFidelityGPSurrogate": SingleFidelityGPSurrogate,
    "MultiFidelityGPSurrogate": MultiFidelityGPSurrogate,
    "SingleFidelityRandomForestSurrogate": SingleFidelityRandomForestSurrogate,
    "SingleFidelityEnsembleSurrogate": SingleFidelityEnsembleSurrogate,
}
surrogate_choice = st.sidebar.selectbox("Surrogate type", list(SURROGATE_OPTIONS.keys()))

model_file = st.sidebar.file_uploader("Surrogate model file (optional, .pt / .pth)", type=["pt", "pth"])

# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Loading domain…")
def load_domain(file_bytes, filename):
    """Parse the uploaded JSON domain file."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        domain, X_names = InputDomain.read_json(tmp.name)
    os.unlink(tmp.name)
    return domain, X_names


@st.cache_data(show_spinner="Loading state…")
def load_state_from_upload(_domain, X_names, file_bytes, filename):
    """Load a State from either HDF5 or CSV."""
    import tempfile

    ext = os.path.splitext(filename)[1].lower()
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        tmp_path = tmp.name

    if ext in (".h5", ".hdf5"):
        state = State.load(tmp_path, Y_scaler=None)
    elif ext == ".csv":
        df = pd.read_csv(tmp_path)
        col_names = df.columns.tolist()

        # Detect columns: first column = index, then X columns by name matching
        x_cols = [c for c in col_names if c in X_names]
        if not x_cols:
            # Fallback: assume index | X (dim columns) | ... | Y (last column)
            dim = _domain.dim
            x_cols = col_names[1 : 1 + dim]

        # Detect S columns (names starting with 's_' or 'S_' or 'fidelity')
        s_cols = [c for c in col_names if c.lower().startswith("s_") or c.lower() == "fidelity"]

        idx_col = col_names[0]
        y_cols = [col_names[-1]]  # Default: last column is Y

        # Everything not index, X, S, or Y is P
        used = {idx_col} | set(x_cols) | set(s_cols) | set(y_cols)
        p_cols = [c for c in col_names if c not in used]

        Is = df[idx_col].values
        Xs = df[x_cols].values
        Ys = df[y_cols].values
        Ps = df[p_cols].values if p_cols else None
        Ss = df[s_cols].values if s_cols else None

        state = State(
            input_domain=_domain,
            index=Is,
            Xs=Xs,
            Ys=Ys,
            Ps=Ps,
            Ss=Ss,
            X_names=x_cols,
            Y_names=y_cols,
            P_names=p_cols if p_cols else None,
            S_names=s_cols if s_cols else None,
        )
    else:
        st.error(f"Unsupported file extension: {ext}")
        st.stop()

    os.unlink(tmp_path)
    return state


def build_dataframe(state):
    """Convert State arrays into a tidy DataFrame for plotting."""
    data = {}
    if state.index is not None:
        for j in range(state.index.shape[1]):
            name = state.index_names[j] if state.index_names else f"index_{j}"
            data[name] = state.index[:, j]
    if state.Xs is not None:
        for j in range(state.Xs.shape[1]):
            name = state.X_names[j] if state.X_names else f"x_{j}"
            data[name] = state.Xs[:, j]
    if state.Ps is not None:
        for j in range(state.Ps.shape[1]):
            name = state.P_names[j] if state.P_names else f"p_{j}"
            data[name] = state.Ps[:, j]
    if state.Ss is not None:
        for j in range(state.Ss.shape[1]):
            name = state.S_names[j] if state.S_names else f"s_{j}"
            data[name] = state.Ss[:, j]
    if state.Ys is not None:
        for j in range(state.Ys.shape[1]):
            name = state.Y_names[j] if state.Y_names else f"y_{j}"
            data[name] = state.Ys[:, j]

    data["row"] = np.arange(len(next(iter(data.values()))))
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Early exit if inputs not loaded
# ---------------------------------------------------------------------------
if domain_file is None or state_file is None:
    st.info("Upload a **Domain JSON** and a **State file** in the sidebar to get started.")
    st.stop()

domain, X_names = load_domain(domain_file.getvalue(), domain_file.name)
state = load_state_from_upload(domain, X_names, state_file.getvalue(), state_file.name)
df = build_dataframe(state)

st.sidebar.success(f"Loaded {state.nsamples} samples, {domain.dim} dimensions")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_scatter, tab_corner, tab_cv, tab_optim = st.tabs(
    ["📈 Scatter", "🔲 Corner Plot", "📊 Cross-Validation", "🎯 Optimal Design"]
)

# ======================================================================
# TAB 1 — Scatter plotter
# ======================================================================
with tab_scatter:
    st.subheader("State Scatter Plotter")

    all_cols = list(df.columns)
    y_default = state.Y_names[0] if state.Y_names else all_cols[-1]
    x_default = "row"

    col1, col2, col3 = st.columns(3)
    with col1:
        x_axis = st.selectbox("X axis", all_cols, index=all_cols.index(x_default))
    with col2:
        y_axis = st.selectbox("Y axis", all_cols, index=all_cols.index(y_default) if y_default in all_cols else 0)
    with col3:
        colour_by = st.selectbox("Colour by", ["None"] + all_cols)

    fig_scatter, ax_scatter = plt.subplots(figsize=(8, 5))
    if colour_by != "None":
        sc = ax_scatter.scatter(
            df[x_axis], df[y_axis], c=df[colour_by], cmap="viridis",
            alpha=0.7, edgecolors="k", lw=0.3,
        )
        fig_scatter.colorbar(sc, ax=ax_scatter, label=colour_by)
    else:
        ax_scatter.scatter(df[x_axis], df[y_axis], alpha=0.7, edgecolors="k", lw=0.3)
    ax_scatter.set_xlabel(x_axis)
    ax_scatter.set_ylabel(y_axis)
    ax_scatter.grid(True, alpha=0.3)
    st.pyplot(fig_scatter)

# ======================================================================
# TAB 2 — Corner plot
# ======================================================================
with tab_corner:
    st.subheader("Corner Plot of Inputs")

    try:
        import corner
    except ImportError:
        st.error("The `corner` package is required. Install with `pip install millefeuille[dashboard]`.")
        st.stop()

    y_col_corner = st.selectbox(
        "Colour by Y column", state.Y_names if state.Y_names else ["y_0"], key="corner_y",
    )
    y_vals = df[y_col_corner].values

    threshold = st.slider(
        "Y threshold (show points above this value)",
        float(y_vals.min()),
        float(y_vals.max()),
        float(y_vals.min()),
        key="corner_thresh",
    )

    mask = y_vals >= threshold
    X_plot = state.Xs[mask]
    c_vals = y_vals[mask]

    if X_plot.shape[0] < 2:
        st.warning("Fewer than 2 points pass the threshold — relax the slider.")
    else:
        labels = state.X_names if state.X_names else [f"x_{i}" for i in range(domain.dim)]

        fig_corner = corner.corner(
            X_plot,
            labels=labels,
            show_titles=True,
            title_fmt=".3g",
            plot_datapoints=True,
            plot_density=True,
            plot_contours=True,
        )
        sm = plt.cm.ScalarMappable(
            cmap="viridis", norm=plt.Normalize(vmin=c_vals.min(), vmax=c_vals.max()),
        )
        sm.set_array([])
        fig_corner.colorbar(sm, ax=fig_corner.axes, label=y_col_corner, shrink=0.6, pad=0.08)
        st.pyplot(fig_corner)
        st.caption(f"Showing {mask.sum()} / {len(mask)} points above threshold {threshold:.4g}")

# ======================================================================
# TAB 3 — k-fold cross-validation
# ======================================================================
with tab_cv:
    st.subheader("k-Fold Cross-Validation")

    SurrogateClass = SURROGATE_OPTIONS[surrogate_choice]
    k_folds = st.number_input("Number of folds", min_value=2, max_value=20, value=5, step=1, key="cv_k")
    run_cv = st.button("Run Cross-Validation")

    if run_cv:
        with st.spinner("Running cross-validation…"):
            Xs, Ys, Is = state.Xs, state.Ys, state.index
            is_multi = state.l_MultiFidelity
            Ss_flat = state.Ss[:, 0] if is_multi else None

            kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
            cv_metrics = []
            fold_figures = []

            progress = st.progress(0)
            for fold_i, (train_idx, test_idx) in enumerate(kf.split(Xs)):
                X_train, X_test = Xs[train_idx], Xs[test_idx]
                Y_train, Y_test = Ys[train_idx], Ys[test_idx]
                I_train = Is[train_idx]

                fold_state = State(
                    state.input_domain,
                    index=I_train,
                    Xs=X_train,
                    Ys=Y_train,
                    Ss=Ss_flat[train_idx] if is_multi else None,
                    fidelity_domain=state.fidelity_domain,
                )

                surrogate = SurrogateClass()

                if model_file is not None and hasattr(surrogate, "load"):
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
                        tmp.write(model_file.getvalue())
                        tmp.flush()
                        surrogate.fit(fold_state)
                        surrogate.load(tmp.name)
                    os.unlink(tmp.name)
                else:
                    surrogate.fit(fold_state)

                fig_fold, (ax_pred, ax_resid) = plt.subplots(1, 2, figsize=(10, 4))

                if is_multi:
                    metrics, results = evaluate_GP_scores_multi_fid(
                        surrogate, fold_state, X_test, Y_test, Ss_flat[test_idx], verbose=False,
                    )
                    y_true_l, y_pred_l, y_std_l = results[:3]
                    y_true_h, y_pred_h, y_std_h = results[3:]

                    ax_pred.errorbar(
                        y_true_l, y_pred_l, yerr=y_std_l, fmt="o", alpha=0.4, c="blue",
                        label=f"Low fid (R²={metrics['low']['R2']:.3f})",
                    )
                    ax_pred.errorbar(
                        y_true_h, y_pred_h, yerr=y_std_h, fmt="o", alpha=0.4, c="red",
                        label=f"High fid (R²={metrics['high']['R2']:.3f})",
                    )
                    resid_l = y_true_l - y_pred_l
                    resid_h = y_true_h - y_pred_h
                    ax_resid.hist(resid_l, bins=20, alpha=0.5, color="blue", label="Low fid")
                    ax_resid.hist(resid_h, bins=20, alpha=0.5, color="red", label="High fid")
                else:
                    metrics, results = evaluate_GP_scores_single_fid(
                        surrogate, fold_state, X_test, Y_test, verbose=False,
                    )
                    y_true, y_pred, y_std = results
                    y_true, y_pred, y_std = map(np.squeeze, (y_true, y_pred, y_std))

                    ax_pred.errorbar(
                        y_true, y_pred, yerr=y_std, fmt="o", alpha=0.4, c="steelblue",
                        label=f"R²={metrics['R2']:.3f}",
                    )
                    resid = y_true - y_pred
                    ax_resid.hist(resid, bins=20, alpha=0.6, color="steelblue")

                all_vals = df[state.Y_names[0]].values if state.Y_names else df.iloc[:, -1].values
                ax_pred.plot([all_vals.min(), all_vals.max()], [all_vals.min(), all_vals.max()], "k--")
                ax_pred.set_xlabel("Truth")
                ax_pred.set_ylabel("Prediction")
                ax_pred.set_title(f"Fold {fold_i + 1}")
                ax_pred.legend()
                ax_pred.grid(True, alpha=0.3)
                ax_resid.set_xlabel("Residual")
                ax_resid.set_ylabel("Frequency")
                ax_resid.set_title(f"Residuals — Fold {fold_i + 1}")
                ax_resid.grid(True, alpha=0.3)

                fig_fold.tight_layout()
                fold_figures.append(fig_fold)
                cv_metrics.append(metrics)
                progress.progress((fold_i + 1) / k_folds)

            for fig_fold in fold_figures:
                st.pyplot(fig_fold)

            st.subheader("Summary")
            if is_multi:
                summary_data = {
                    "Fold": list(range(1, k_folds + 1)),
                    "R² (low)": [m["low"]["R2"] for m in cv_metrics],
                    "MAE (low)": [m["low"]["MAE"] for m in cv_metrics],
                    "RMSE (low)": [m["low"]["RMSE"] for m in cv_metrics],
                    "R² (high)": [m["high"]["R2"] for m in cv_metrics],
                    "MAE (high)": [m["high"]["MAE"] for m in cv_metrics],
                    "RMSE (high)": [m["high"]["RMSE"] for m in cv_metrics],
                }
            else:
                summary_data = {
                    "Fold": list(range(1, k_folds + 1)),
                    "R²": [m["R2"] for m in cv_metrics],
                    "MAE": [m["MAE"] for m in cv_metrics],
                    "RMSE": [m["RMSE"] for m in cv_metrics],
                }

            df_summary = pd.DataFrame(summary_data)
            st.dataframe(df_summary.style.format(precision=4), use_container_width=True)

            if is_multi:
                r2_low = [m["low"]["R2"] for m in cv_metrics]
                r2_high = [m["high"]["R2"] for m in cv_metrics]
                st.metric("Mean R² (low fidelity)", f"{np.mean(r2_low):.4f} ± {np.std(r2_low):.4f}")
                st.metric("Mean R² (high fidelity)", f"{np.mean(r2_high):.4f} ± {np.std(r2_high):.4f}")
            else:
                r2_vals = [m["R2"] for m in cv_metrics]
                st.metric("Mean R²", f"{np.mean(r2_vals):.4f} ± {np.std(r2_vals):.4f}")


# ======================================================================
# TAB 4 — Optimal design landscape
# ======================================================================
with tab_optim:
    st.subheader("Optimal Design Landscape")
    st.markdown(
        "For each input dimension, hold that variable at a fixed value "
        "and optimise the surrogate posterior mean over the remaining "
        "dimensions. This reveals how the optimum depends on each input."
    )

    n_points = st.slider("Number of points along fixed axis", 10, 100, 40, key="optim_npts")
    run_optim = st.button("Run Optimal-Design Analysis")

    if run_optim:
        SurrogateClass = SURROGATE_OPTIONS[surrogate_choice]
        surrogate = SurrogateClass()

        with st.spinner("Fitting surrogate…"):
            if model_file is not None and hasattr(surrogate, "load"):
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
                    tmp.write(model_file.getvalue())
                    tmp.flush()
                    surrogate.fit(state)
                    surrogate.load(tmp.name)
                os.unlink(tmp.name)
            else:
                surrogate.fit(state)

        lb = torch.zeros(domain.dim, dtype=torch.double)
        ub = torch.ones(domain.dim, dtype=torch.double)
        bounds = torch.stack([lb, ub])

        xfixeds = np.linspace(0.0, 1.0, n_points)
        labels = state.X_names if state.X_names else [f"x_{i}" for i in range(domain.dim)]

        n_cols = min(domain.dim, 4)
        n_rows = int(np.ceil(domain.dim / n_cols))
        fig_optim, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), squeeze=False)

        progress = st.progress(0)
        for i in range(domain.dim):
            y_fixeds = []
            y_stds = []

            for xf in xfixeds:
                Xopt, current_value = optimize_acqf(
                    acq_function=PosteriorMean(surrogate.model),
                    bounds=bounds,
                    q=1,
                    num_restarts=10,
                    raw_samples=512,
                    fixed_features={i: float(xf)},
                )
                y_fixeds.append(current_value.item())
                Xopt_real = state.inverse_transform_X(Xopt.detach().cpu().numpy())
                pred = surrogate.predict(state, Xopt_real)
                y_stds.append(pred["std"].item())

            y_fixeds = np.array(y_fixeds)
            y_stds = np.array(y_stds)

            y_fixeds, y_stds = state.inverse_transform_Y(y_fixeds, y_stds)
            y_fixeds = np.asarray(y_fixeds).flatten()
            y_stds = np.asarray(y_stds).flatten()

            x_real = domain.b_low[i] + (domain.b_up[i] - domain.b_low[i]) * xfixeds

            ax = axs.flat[i]
            ax.plot(x_real, y_fixeds, c="steelblue")
            ax.fill_between(x_real, y_fixeds - y_stds, y_fixeds + y_stds, alpha=0.25, fc="steelblue")
            ax.set_xlabel(labels[i])
            ax.set_ylabel("Y*")
            ax.set_xlim(x_real[0], x_real[-1])
            ax.grid(True, alpha=0.3)

            progress.progress((i + 1) / domain.dim)

        for j in range(domain.dim, len(axs.flat)):
            axs.flat[j].set_visible(False)

        fig_optim.suptitle("Optimal Y* vs. each input (others optimised)", fontsize=13)
        fig_optim.tight_layout()
        st.pyplot(fig_optim)
