## README

### Overview

This example demonstrates `mille-feuille`'s **asynchronous scheduling** capabilities using `run_async_loop`. Instead of the synchronous batch-wait-analyse pattern used in the `loops/` example, the async loop continuously launches and collects simulation evaluations to maximise core utilisation on a single compute node.

The workflow uses a two-phase sampling strategy:
1. **Random phase** — Sobol quasi-random samples fill the design space.
2. **Threshold phase** — once enough data is collected, a `RandomForestEnsembleModel` surrogate is trained and `probabilistic_threshold_filter` biases sampling toward promising regions.

> **Single-node restriction:** This example assumes all MPI jobs run on one compute node. The `MPIScheduler` launches jobs with plain `mpiexec` — no host files or multi-node placement. For multi-node PBS scheduling, see the `loops/` example.

### File Descriptions

###### Utility Scripts

**Simulator.py**
Defines simulation interface:

`Simulator`: A generic single-fidelity executable simulator that manages Fortran namelist-based input generation, MPI job launching via a scheduler, and post-processing of simulation outputs. Provides template methods for reading diagnostics and computing objectives.

**Scheduler.py**
Defines scheduler interface:

`MPIScheduler`: A single-node MPI scheduler that launches jobs via `mpiexec -n <nproc>`. Each job's stdout/stderr is captured in per-run log files. Thread-safe for use with the async loop's `ThreadPoolExecutor`.

**Utils.py**
Provides utility functions including: reading training data from CSV, reading domain definitions from JSON, and safe directory creation/removal.

###### Execution Scripts

**AsyncSampler.py**
Main entry point. Reads the domain from `domain.json`, configures a `ResourceManager` with the available cores, and runs the full asynchronous loop via `run_async_loop`. Implements `RandomThenThresholdCandidateGenerator` — a custom `CandidateGenerator` subclass that switches from Sobol sampling to surrogate-guided probabilistic threshold sampling after a configurable number of initial evaluations.

###### Configuration

**domain.json**
JSON domain definition specifying input parameter names, lower/upper bounds, and discretisation steps.
