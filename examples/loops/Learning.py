import millefeuille as mf
import numpy as np
from Scheduler import PBSMPIScheduler, get_PBS_hosts
from Simulator import Simulator
from Utils import Uniform, get_training_data, read_domainfile

EXECUTABLE = "path_to_your_executable"

if __name__ == "__main__":
    # Parameters
    domain_file = "Domain.json"
    training_file = "Initialise_DB.csv"
    output_file = "Learning_DB.csv"

    domain, X_names = read_domainfile(domain_file)
    # Read the output file to get training data
    Is, Xs, Ys, Ps, X_names, Y_names, P_names = get_training_data(training_file, domain.dim)

    # Initialise state
    state = mf.State(
        input_domain=domain, index=Is, Xs=Xs, Ys=Ys, Ps=Ps, X_names=X_names, Y_names=Y_names, P_names=P_names
    )
    # Initialise surrogate
    surrogate = mf.SingleFidelityGPSurrogate()

    base_paths = {
        "exe": f"{EXECUTABLE}",
        "run_dir": "./runs/",
        "input": "./input_base.f90",
        "inputs_dir": "./input_decks/",
    }
    io_params = {}

    # Initialise simulator
    batched_simulator = Simulator(base_paths, io_params)

    nproc = batched_simulator.nproc

    # Batching options and initialise scheduler
    l_batch = True
    output_dir = "./screen_outputs/"
    Nbatch, hosts = get_PBS_hosts(l_batch, nproc)
    scheduler = PBSMPIScheduler(output_dir, hosts)

    # Active learning parameters
    import sys

    # Super batch number from command line argument
    if len(sys.argv) < 2:
        array_job_num = 0
    else:
        array_job_num = int(sys.argv[-1])
    # Number of learning runs
    Nlearningruns = 128
    # Initial samples for adaptive sampling
    initial_samples = 64
    # Probability limit for adaptive sampling
    prob_lim = 0.25
    # Surrogate threshold for adaptive sampling
    surrogate_threshold = 0.9 * state.best_value
    # Whether to update threshold after each run
    update_threshold = False
    # Retrain surrogate model after N new samples
    retrain_surrogate_num = 4
    # Sampler for adaptive sampling
    sampler = Uniform(domain.dim)

    # Starting index for this batch
    current_iter = np.amax(Is) + array_job_num * Nlearningruns * Nbatch
    # Main loop for learning runs
    for i in range(Nlearningruns):
        # Train surrogate
        if i * Nbatch % retrain_surrogate_num == 0:
            print("Fitting surrogate...")
            surrogate.fit(state)
            print("Surrogate fit!")

        # Adaptive sampling
        sum_mask = 0
        initial_samples = initial_samples // 2
        while sum_mask < Nbatch:
            initial_samples *= 2
            random_draws = prob_lim + (1 - prob_lim) * np.random.rand(initial_samples)
            x_all, y_pred, prob, mask = mf.probabilistic_threshold_sampling(
                domain,
                state,
                sampler,
                surrogate,
                initial_samples,
                surrogate_threshold,
                random_draws=random_draws,
            )
            sum_mask = mask.sum()

        # Select Nbatch from masked x
        X_mask = x_all[mask, :]
        X_next = X_mask[np.random.choice(X_mask.shape[0], size=Nbatch, replace=False), :]
        index_next = current_iter + np.arange(Nbatch) + 1

        # Run batches of Chimera
        P_next, Y_next = batched_simulator(index_next, X_next, scheduler)

        state.update(index_next, X_next, Y_next, P_next=P_next)

        if update_threshold:
            surrogate_threshold = 0.9 * state.best_value

        current_iter += Nbatch

        state.to_csv(output_file)
