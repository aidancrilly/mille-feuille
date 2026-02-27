import millefeuille as mf
import numpy as np
from Simulator import Simulator

from Utils import read_domainfile

EXECUTABLE = "path_to_your_executable"

if __name__ == "__main__":
    # Parameters
    inputs_save_file = "Initialise_inputs.csv"
    design_file = "Domain.json"
    training_file = "Initialise_DB.csv"

    domain, X_names = read_domainfile(design_file)

    data = np.loadtxt(inputs_save_file, delimiter=",")
    Is, Xs = data[:, :1], data[:, 1:]

    # Execution
    base_paths = {
        "exe": f"{EXECUTABLE}",
        "run_dir": "./runs/",
        "input": "./input_base.nml",
        "inputs_dir": "./input_decks/",
    }
    io_params = {}

    # Initialise simulator
    batched_simulator = Simulator(base_paths, io_params)

    Ps, Ys = batched_simulator.postprocess(Is, Xs)

    # Set up variable names for state csv
    P_names = None
    Y_names = None

    state = mf.State(
        input_domain=domain, index=Is, Xs=Xs, Ys=Ys, Ps=Ps, X_names=X_names, Y_names=Y_names, P_names=P_names
    )

    state.to_csv(training_file)
