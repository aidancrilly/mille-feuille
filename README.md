
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
pip install .[dev]
```

Requires **Python ≥ 3.11**. Core dependencies (`botorch`, `gpytorch`, `numpy`, `scipy`, `h5py`, `scikit‑learn` …) are pulled in automatically.

---

## 🚀 Quick‑start

```python
import numpy as np
from scipy.stats.qmc import LatinHypercube
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

# Initial random sampling
nsims = 10
sampler = LatinHypercube(domain.dim)
index = np.arange(nsims,dtype=int)
Xs = generate_initial_sample(domain,sampler,nsims)
Ys = -np.vecdot(Xs,Xs,axis=1)

# Package into millefeuille State
state = State(
	input_domain=domain,
	index=index,
	Xs=Xs,
	Ys=Ys
)

# 2. Optimiser call ---------------------------------------------------------
X_next = suggest_next_locations(
    batch_size=2,
    state=state,
    surrogate=SingleFidelityGPSurrogate(),
    acq_function="qLogExpectedImprovement"
)
print(X_next)    # candidate points in original scale
```

Plug in your own simulator to evaluate `X_next`, update the `State`, and repeat.

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
