"""Utility helpers for the asynchronous scheduling example."""

import json
import os
import shutil

import millefeuille as mf
import numpy as np
import pandas as pd
from scipy.stats import qmc


def get_training_data(output_file, dim):
    """Load training data from a CSV produced by ``State.to_csv``.

    Returns:
        (Is, Xs, Ys, Ps, X_names, Y_names, P_names)
    """
    if not os.path.isfile(output_file):
        raise FileNotFoundError(f"Output file {output_file} does not exist.")

    df = pd.read_csv(output_file, index_col=0)
    Is = df.index.values
    Xs = df.iloc[:, :dim].values
    Ps = df.iloc[:, dim:-1].values
    Ys = df.iloc[:, -1:].values
    names = df.keys().tolist()
    X_names = names[:dim]
    P_names = names[dim:-1]
    Y_names = names[-1:]

    return Is, Xs, Ys, Ps, X_names, Y_names, P_names


def read_domainfile(file):
    """Read a JSON domain definition and return ``(InputDomain, X_names)``."""
    with open(file, "r") as json_file:
        json_dict = json.load(json_file)

    n_x = len(json_dict["params"]["names"])
    b_low = np.array(json_dict["params"]["lower_bounds"])
    b_up = np.array(json_dict["params"]["upper_bounds"])
    steps = np.array(json_dict["params"]["steps"])

    domain = mf.InputDomain(
        dim=n_x,
        b_low=b_low,
        b_up=b_up,
        steps=steps,
    )

    X_names = json_dict["params"]["names"]

    return domain, X_names


def safe_rmdir(directory):
    """Remove *directory* and all contents, ignoring errors."""
    try:
        if os.path.exists(directory) and os.path.isdir(directory):
            shutil.rmtree(directory)
    except OSError as err:
        print(err)


def safe_mkdir(directory):
    """Create *directory* (and parents) if it does not already exist."""
    os.makedirs(directory, exist_ok=True)
