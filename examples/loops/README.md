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

All execution scripts use `mille‑feuille`’s composable **candidate generators** to produce inputs for evaluation.  Generators are passed to `run_generator_loop` which handles the generate → evaluate → update cycle.

**Initialise.py**
Generates a set of initial samples using `RandomCandidateGenerator` with a Sobol sampler, and prepares the corresponding input files for simulations. Use **PostInitialise.py** to collect data after running array job on sample input files.

**Learning.py**
Implements an active learning loop using `ThresholdCandidateGenerator`.  Trains a surrogate model (Gaussian Process) and adaptively selects new samples based on prediction uncertainty via `run_generator_loop`.

**Optimise.py**
Runs Bayesian optimisation using `BayesianOptimisationGenerator` with the `qLogExpectedImprovement` acquisition function from BoTorch. Uses `run_generator_loop` to iterate the surrogate-based optimisation cycle.
