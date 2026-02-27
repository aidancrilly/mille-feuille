Quick-start
===========

The example below shows the minimal workflow for a **single-fidelity**
Bayesian optimisation run using a GP surrogate.

.. code-block:: python

   import numpy as np
   from scipy.stats import qmc
   from botorch.acquisition import qExpectedImprovement

   from millefeuille.domain import InputDomain
   from millefeuille.state import State
   from millefeuille.surrogate import SingleFidelityGPSurrogate
   from millefeuille.initialise import generate_initial_sample
   from millefeuille.utils import run_Bayesian_optimiser
   from millefeuille.simulator import PythonSimulator

   # 1. Define the input domain
   domain = InputDomain(
       dim=1,
       b_low=np.array([0.0]),
       b_up=np.array([1.0]),
       steps=np.array([0.0]),   # 0 = continuous
   )

   # 2. Collect initial samples
   sampler = qmc.LatinHypercube(d=domain.dim, seed=42)
   X_init, n_init = generate_initial_sample(domain, sampler, initial_samples=5)

   # 3. Evaluate your simulator on the initial samples
   # (replace with your own function)
   def my_function(indices, Xs, Ss=None):
       return None, -((6 * Xs - 2) ** 2 * np.sin(12 * Xs - 4))

   class MySim(PythonSimulator):
       def __call__(self, indices, Xs, Ss=None):
           return my_function(indices, Xs, Ss)

   simulator = MySim()
   indices_init = np.arange(n_init)
   _, Y_init = simulator(indices_init, X_init)

   # 4. Create the initial state
   state = State(
       input_domain=domain,
       index=indices_init,
       Xs=X_init,
       Ys=Y_init,
   )

   # 5. Define a factory that returns the acquisition function
   def make_acq(surrogate, state):
       return qExpectedImprovement(
           model=surrogate.model,
           best_f=state.best_value_transformed,
       )

   # 6. Run the optimisation loop
   surrogate = SingleFidelityGPSurrogate()
   state = run_Bayesian_optimiser(
       Nsamples=10,
       batch_size=1,
       generate_acq_function=make_acq,
       state=state,
       surrogate=surrogate,
       simulator=simulator,
   )

   print("Best value found:", state.best_value)

For multi-fidelity and ensemble surrogate examples, see the ``examples/``
directory in the repository.
