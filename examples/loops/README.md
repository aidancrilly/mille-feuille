## README

### File Descriptions

###### Utility Scripts

**Simulator.py**
Defines simulation interface:

`Simulator`: Manages simulation runs including input generation and postprocessing of results.

**Scheduler.py**
Defines scheduler interface:

`PBSMPIScheduler`: Handles MPI job scheduling across multiple hosts, managing output and error logs within a [PBS](https://www.openpbs.org/) environment.

**Utils.py**
Provides utility functions including: reading in data and domains, and create and deleting directories.

###### Execution Scripts

**Initialise.py**
Initializes the simulation process by generating a set of initial samples using a Sobol sequence, and prepares the corresponding input files for simulations. Use **PostInitialise.py** to collect data after running array job on sample input files.

**Learning.py**
Implements an active learning loop using a probabilistic thresholding strategy. Trains a surrogate model (Gaussian Process) and adaptively selects new samples based on prediction uncertainty and performance.

**Optimise.py**
Runs Bayesian optimization using the qLogExpectedImprovement acquisition function from BoTorch. Selects high-potential samples for evaluation based on the current surrogate model and updates the optimization state.
