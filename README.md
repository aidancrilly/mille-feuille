# mille-feuille

![workflow badge](https://github.com/aidancrilly/mille-feuille/actions/workflows/run_tests.yaml/badge.svg)

<center><img src="https://github.com/aidancrilly/mille-feuille/blob/main/logo/MF_logo.webp" width="200" title="mille-feuille" alt="mille-feuille" align="middle"/></center>

`mille‑feuille` acts as an orchestrator when running sampling, learning and optimisation loops against expensive MPI-parallelised HPC codes. For optimisation, `mille‑feuille` is a thin wrapper on top of [**BoTorch**](https://botorch.org/), providing the necessary interface between simulators, surrogates and optimisers.

> **Status:** early days – very much a work in progress.

Used in the following publications:

- [Automated simulation-based design via multi-fidelity active learning and optimisation for laser direct drive implosions, Crilly et al.](https://arxiv.org/abs/2508.20878)

---

## 🔧 Install

Pip:

```bash
# Development head
pip install git+https://github.com/aidancrilly/mille-feuille.git
```

Or clone repo and install dev environment:

```bash
git clone https://github.com/aidancrilly/mille-feuille.git
cd mille-feuille
pip install -e .[dev]
python dev_fetch_deps.py # Grabs header file needed by C++ test script
```

Requires **Python ≥ 3.11**. Core dependencies (`botorch`, `gpytorch`, `numpy`, `scipy`, `scikit‑learn` …) are pulled in automatically.

---

## 🚀 Quick‑start

Take a look at the examples directory and sub-directories within

* *test_executables*: simple (fortran90 and C++) examples implemented for the test suite.
* *loops*: example scripts which use `mille‑feuille` to perform sampling, learning and optimisation tasks. This example includes a template for a simulator with namelist based input and scheduling within a PBS environment.
* *async_scheduling*: **(preliminary)** asynchronous scheduling example using `run_async_loop` to continuously launch and collect simulation evaluations, maximising core utilisation on a single compute node.

## Core components

### Domains, States and Surrogates

`mille‑feuille` implements the following containers:

1. *InputDomain* holds the bounded input domain which can be a mix of continuous and discrete dimensions. Domains can be created from a JSON file with `InputDomain.read_json(filepath)`.
2. *FidelityDomain* holds information regarding the degrees of simulation fidelity.
3. *State* holds the necessary data taken from simulation samples: indices (Is), inputs (Xs), output parameters (Ps), fidelities (Ss) and objectives (Ys). State now supports **empty initialisation** (all arrays `None`) with lazy naming on the first update — useful for workflows where no data exists at construction time. Persistence uses **SQLite** (`state.save()` / `State.load()`) for robust, append-friendly storage.

These classes hold the necessary information to train surrogate models. `mille‑feuille` has a number of abstract base classes as well as concrete examples of surrogate models including **Gaussian Processes** (using GPyTorch and BOTorch), **Neural Network Ensembles** (using PyTorch and BOTorch) and **Random Forest ensembles** (via scikit-learn's `RandomForestRegressor`).

### Simulators and Schedulers

`mille‑feuille` implements abstract base classes to interact with generic executable simulators. These follow a simple design pattern that simulators are expected to follow:

1. External input files are written based on the candidate X values
2. Batches of the (mpiexec'ed) simulator are launched via a Scheduler
3. The simulator writes output files which must be post-processed to extract useful information (P) and the objective function value (Y)
4. (Optionally) A clean up of the Simulator and Schedular output files is performed

Take a look at the examples and the tests to see implementations of Simulators and Schedulers.

> Preliminary **asynchronous scheduling** support is available — see the [async_scheduling example](examples/async_scheduling/).

The async system introduces several supporting components:

| Component | Purpose |
|---|---|
| `FidelityConfig` | Per-fidelity core count and reservation settings |
| `Task` | Lightweight `(index, X, S)` task descriptor |
| `ResourceManager` | Thread-safe core accounting (allocate / release / utilisation) |
| `AsyncScheduler` | Main scheduling loop with reservation + backfill strategy |

### Run loops

`mille‑feuille` provides three high-level loop helpers:

* `run_Bayesian_optimiser` — synchronous batch loop (sample → learn → optimise → repeat).
* `run_generator_loop` — synchronous loop driven by any `CandidateGenerator`, decoupling the candidate strategy from the evaluation loop.
* `run_async_loop` — asynchronous execution, `AsyncScheduler` continuously launches and collects simulations via a `ThreadPoolExecutor` and supplied `CandidateGenerator`.

### Fixed-feature optimisation

`suggest_next_locations` (and the underlying `generate_batch`) accepts a `fixed_features` dictionary to pin specific input dimensions to fixed values during acquisition-function optimisation. Values are automatically transformed from real to normalised units.

### Candidate Generators

`mille‑feuille` provides an interchangable system of **candidate generators** that produce the next batch of inputs to evaluate.  Every generator implements the same `CandidateGenerator` interface and returns `(indices, Xs, Ss)`, so they can be plugged into `run_generator_loop` or the async scheduler interchangeably.

#### Base generators

Base generators produce candidates from scratch:

| Generator | Strategy |
|---|---|
| `RandomCandidateGenerator` | Uniform / QMC sampling over the input domain |
| `BayesianOptimisationGenerator` | Surrogate + acquisition-function optimisation |
| `ThresholdCandidateGenerator` | Probabilistic threshold sampling (draws pool → surrogate prediction → stochastic filter) |
| `SurrogateThresholdCandidateGenerator` | Deterministic surrogate threshold sampling (predicted mean > threshold) |

#### Wrapper generators

Wrapper generators take another generator as input and refine its output:

| Generator | Effect |
|---|---|
| `GreedyExclusionGenerator` | PCA-normalised proximity-based exclusion to avoid clustered candidates.

## Additional Info

CI makes use of [cached IntelOneAPI install by scivision](https://gist.github.com/scivision/b22455e3322826a1c385d5d4b1a8d25e)

## ✉️ Contact

**Aidan Crilly** · [ac116@ic.ac.uk](mailto\:ac116@ic.ac.uk)

---

Licensed under MIT © 2025
