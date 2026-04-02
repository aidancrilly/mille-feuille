import millefeuille as mf
from Scheduler import PBSMPIScheduler, get_PBS_hosts
from Simulator import Simulator

# Import utils (N.B. local utils file, not millefeuille.utils)
from Utils import get_training_data, read_domainfile

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
        "input": "./input_base.nml",
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
    Nlearningruns = 128
    surrogate_threshold = 0.9 * state.best_value
    pool_size = 256
    prob_lim = 0.25

    # Sampler for adaptive sampling
    from Utils import Uniform

    sampler = Uniform(domain.dim)

    # ---- Build a composable generator ----
    # ThresholdCandidateGenerator draws a pool, predicts with the surrogate,
    # and filters probabilistically.  min_probability biases draws so that
    # only high-confidence candidates pass the filter.
    generator = mf.ThresholdCandidateGenerator(
        domain=domain,
        sampler=sampler,
        surrogate=surrogate,
        threshold_value=surrogate_threshold,
        pool_size=pool_size,
        refit_surrogate=True,
        min_probability=prob_lim,
    )

    # Run the generate-evaluate loop
    state = mf.run_generator_loop(
        Nsamples=Nlearningruns,
        batch_size=Nbatch,
        generate_candidates=generator,
        state=state,
        simulator=batched_simulator,
        scheduler=scheduler,
        csv_name=output_file,
    )
