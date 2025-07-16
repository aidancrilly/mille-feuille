# mille-feuille

`mille‑feuille` is a thin layer on top of [**BoTorch**](https://botorch.org/) that adds the plumbing you need to run optimisation loops against expensive HPC codes—locally while you prototype, or scaled out on a cluster when the time comes.

> **Status:** early days – very much a work in progress

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
```

Requires **Python ≥ 3.11**. Core dependencies (`botorch`, `gpytorch`, `numpy`, `scipy`, `h5py`, `scikit‑learn` …) are pulled in automatically.

---

## 🚀 Quick‑start

### States and Surrogates

```python
import numpy as np

from scipy.stats.qmc import LatinHypercube

from botorch.acquisition import qUpperConfidenceBound

from millefeuille.domain import InputDomain
from millefeuille.state import State
from millefeuille.surrogate import SingleFidelityGPSurrogate
from millefeuille.optimise import suggest_next_locations
from millefeuille.initialise import generate_initial_sample

# 1. Problem setup ----------------------------------------------------------
domain = InputDomain(
    dim=2,
    b_low=np.array([-5.0, -5.0]),
    b_up=np.array([5.0, 5.0]),
    steps=np.array([0.0, 0.0])   # both dims continuous
)

# Function to be maximised
def f_optim(x):
    return -np.vecdot(x,x,axis=1)

# 2. Create initial training data -------------------------------------------

# Initial random sampling
nsims = 10
sampler = LatinHypercube(domain.dim)
index = np.arange(nsims,dtype=int)
Xs, _ = generate_initial_sample(domain,sampler,nsims)
Ys = f_optim(Xs)

# 3. Set up mille-feuille state & surrogate ---------------------------------
# Package into millefeuille State
state = State(
	input_domain=domain,
	index=index,
	Xs=Xs,
	Ys=Ys
)

# Create surrogate model based on state
surrogate = SingleFidelityGPSurrogate()
surrogate.init(state)

# 4. Optimiser call ---------------------------------------------------------
# Use botorch acquistion functions
acq_function = qUpperConfidenceBound(surrogate.model,beta=0.5)

X_next = suggest_next_locations(
    batch_size=2,
    state=state,
    surrogate=surrogate,
    acq_function=acq_function
)
print(X_next)    # candidate points in original scale
```

Plug in your own simulator to evaluate `X_next`, update the `State`, and repeat.

### Simulators and Schedulers

`mille‑feuille` implements abstract base classes to interact with generic executable simulators. These follow a simple design pattern that simulators are expected to follow:

1. External input files are written based on the candidate X values
2. Batches of the (mpiexec'ed) simulator are launched via a Scheduler
3. The simulator writes output files which must be post-processed to extract useful information (P) and the objective function value (Y)
4. (Optionally) A clean up of the Simulator and Schedular output files is performed

Take a look at the simple (fortran90 and C++) examples implemented in the test suite in `conftest.py` and /tests/test_exe/. The C++ example makes use of the [nlohmann JSON](https://github.com/nlohmann/json) header file.

---

## 🤝 Relationship to BoTorch

`mille‑feuille` is *additive*: all modelling and acquisition is delegated to BoTorch. What we provide is

- **Thin wrappers** around BoTorch models so you can swap simulators and surrogates without touching the optimiser loop.
- **State and simulator bookkeeping** streamlining the interaction between HPC codes and BO.
- **Job scheduling adapters** so the same simulator class can be adapted to different HPC systems.

When you need full flexibility you can always drop down and call BoTorch directly.

---

## 🛠️ Roadmap / work in progress

- **Alternative surrogates** – initial experiments with NN-ensemble based surrogates.
- **Schedulers** – reference Slurm & PBS implementations.
- **Examples** – Show .`Simulator` Classes for open source HPC physics codes.

---

## ✉️ Contact

**Aidan Crilly** · [ac116@ic.ac.uk](mailto\:ac116@ic.ac.uk)

---

Licensed under MIT © 2025
