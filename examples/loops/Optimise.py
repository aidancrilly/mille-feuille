import millefeuille as mf
from botorch.acquisition import qLogExpectedImprovement
from Scheduler import PBSMPIScheduler, get_PBS_hosts
from Simulator import Simulator

from Utils import get_training_data, read_domainfile

EXECUTABLE = "path_to_your_executable"

if __name__ == "__main__":
    # Parameters
    domain_file = "Domain.json"
    training_file = "Learning_DB.csv"
    output_file = "Optimise_DB.csv"

    # Execution
    base_paths = {
        "exe": f"{EXECUTABLE}",
        "run_dir": "./runs/",
        "input": "./input_base.nml",
        "inputs_dir": "./input_decks/",
    }
    io_params = {}

    domain = read_domainfile(domain_file)
    # Read the output file to get training data
    Is, Xs, Ys, Ps, X_names, Y_names, P_names = get_training_data(training_file, domain.dim)

    # Initialise state
    state = mf.State(
        input_domain=domain, index=Is, Xs=Xs, Ys=Ys, Ps=Ps, X_names=X_names, Y_names=Y_names, P_names=P_names
    )
    # Initialise surrogate
    surrogate = mf.SingleFidelityGPSurrogate()
    # Initialise simulator
    batched_simulator = Simulator(base_paths, io_params)

    nproc = batched_simulator.nproc

    # Batching options and initialise scheduler
    l_batch = False
    output_dir = "./screen_outputs/"
    Nbatch, hosts = get_PBS_hosts(l_batch, nproc)
    scheduler = PBSMPIScheduler(output_dir, hosts)

    Nsamples = 256

    def generate_LEI_acq(surrogate, state):
        acq_function = qLogExpectedImprovement(surrogate.model, best_f=state.best_value_transformed)
        return acq_function

    mf.run_Bayesian_optimiser(
        Nsamples, Nbatch, generate_LEI_acq, state, surrogate, batched_simulator, scheduler, csv_name=output_file
    )
