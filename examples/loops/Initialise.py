import millefeuille as mf
import numpy as np
from scipy.stats.qmc import Sobol
from Simulator import Simulator

# Import utils (N.B. local utils file, not millefeuille.utils)
from Utils import read_domainfile

EXECUTABLE = "path_to_your_executable"

if __name__ == "__main__":
    # Parameters
    inputs_save_file = "Initialise_inputs.csv"
    design_file = "Domain.json"
    # Peform initial_runs_mult * # of input parameter samples
    initial_runs_mult = 10
    current_iter = 0

    domain, _ = read_domainfile(design_file)

    # Make sure divisible by Nbatch and greater than initial_samples
    initial_samples = initial_runs_mult * domain.dim

    sampler = Sobol(d=domain.dim, scramble=True)
    if current_iter > 0:
        _ = sampler.fast_forward(current_iter)

    # ---- Use a RandomCandidateGenerator to draw the initial sample ----
    generator = mf.RandomCandidateGenerator(domain=domain, sampler=sampler)

    log2sample = int(np.ceil(np.log2(initial_samples)))
    n_initial = 2**log2sample
    index_next, X_next, _ = generator(
        # A minimal State with no data yet
        mf.State(input_domain=domain, index=np.array([-1])),
        n_initial,
    )
    index_next = np.arange(current_iter, current_iter + n_initial)

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

    batched_simulator.prepare_inputs(index_next, X_next)

    #############################################################################
    # Inputs now available to be submitted as array job, eg qsub -J 0-[X] ...
    #############################################################################

    # Save inputs to file for PostInitialise
    np.savetxt(inputs_save_file, np.c_[index_next, X_next], delimiter=",")
