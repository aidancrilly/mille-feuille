# mille-feuille

![workflow badge](https://github.com/aidancrilly/mille-feuille/actions/workflows/run_tests.yaml/badge.svg)

<center><img src="https://github.com/aidancrilly/mille-feuille/blob/main/logo/MF_logo.webp" width="200" title="mille-feuille" alt="mille-feuille" align="middle"/></center>

`mille‑feuille` acts as an orchestrator when running sampling, learning and optimisation loops against expensive MPI-parallelised HPC codes. For optimisation, `mille‑feuille` is a thin wrapper on top of [**BoTorch**](https://botorch.org/), providing the necessary interface between simulators, surrogates and optimisers.

> **Status:** early days – very much a work in progress

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

Requires **Python ≥ 3.11**. Core dependencies (`botorch`, `gpytorch`, `numpy`, `scipy`, `h5py`, `scikit‑learn` …) are pulled in automatically.

---

## 🚀 Quick‑start

Take a look at the examples directory and sub-directories within:

* *test_executables*: simple (fortran90 and C++) examples implemented for the test suite.
* *loops*: example scripts which use `mille‑feuille` to perform sampling, learning and optimisation tasks. This example includes a template for a simulator with namelist based input and scheduling within a PBS environment.

## Core components

### Domains, States and Surrogates

`mille‑feuille` implements the following containers:

1. *InputDomain* holds the bounded input domain which can be a mix of continuous and discrete dimension
2. *FidelityDomain* holds information regarding the degrees of simulation fidelity.
3. *State* holds the necessary data taken from simulation samples: indices (Is), inputs (Xs), output parameters (Ps), fidelities (Ss) and objectives (Ys)

These classes hold the necessary information to train surrogate models. `mille‑feuille` has a number of abstract base classes as well as concrete examples of surrogate models including Gaussian Processes (using GPyTorch and BOTorch) and Neural Network Ensembles (using PyTorch and BOTorch).

### Simulators and Schedulers

`mille‑feuille` implements abstract base classes to interact with generic executable simulators. These follow a simple design pattern that simulators are expected to follow:

1. External input files are written based on the candidate X values
2. Batches of the (mpiexec'ed) simulator are launched via a Scheduler
3. The simulator writes output files which must be post-processed to extract useful information (P) and the objective function value (Y)
4. (Optionally) A clean up of the Simulator and Schedular output files is performed

Take a look at the examples and the tests to see implementations of Simulators and Schedulers.

---

## Additional Info

CI makes use of [cached IntelOneAPI install by scivision](https://gist.github.com/scivision/b22455e3322826a1c385d5d4b1a8d25e)

## ✉️ Contact

**Aidan Crilly** · [ac116@ic.ac.uk](mailto\:ac116@ic.ac.uk)

---

Licensed under MIT © 2025
