"""
Asynchronous random-then-threshold sampling with an external executable.

This script demonstrates mille-feuille's ``run_async_loop`` driving an
MPI-based external simulator via the ``MPIScheduler``.

**IMPORTANT — single-node restriction**
  This example assumes all MPI jobs run on a single compute node.  The
  ``MPIScheduler`` in ``Scheduler.py`` launches each job with a plain
  ``mpiexec -n <nproc> <exe> <input>`` command.  No host files or
  multi-node placement is performed.  If you need multi-node scheduling,
  refer to the ``loops/`` example which uses ``PBSMPIScheduler`` with
  PBS host-file support.

Workflow:
  1. Read the domain definition from a JSON file.
  2. Create an empty ``State``.
  3. Instantiate a ``CandidateGenerator`` that switches strategy:
     - Until *n_randomsamples* evaluations have completed, draw Sobol
       quasi-random points.
     - After that, train a ``RandomForestEnsembleModel`` surrogate and
       use probabilistic threshold sampling to focus on promising
       regions.
  4. Hand everything to ``run_async_loop`` which manages concurrent MPI
     launches, collects results, and triggers re-generation of
     candidates as evaluations complete.

Usage:
    python AsyncSampler.py
"""

from millefeuille.asynch import run_async_loop
from millefeuille.domain import InputDomain
from millefeuille.generators import CandidateGenerator, probabilistic_threshold_filter
from millefeuille.simulator import ResourceManager
from millefeuille.state import State
from millefeuille.surrogate import RandomForestEnsembleModel
from Scheduler import MPIScheduler
from scipy.stats.qmc import Sobol
from Simulator import Simulator

EXECUTABLE = "path_to_your_executable"


# ── Candidate generator ──────────────────────────────────────────────────────


class RandomThenThresholdCandidateGenerator(CandidateGenerator):
    """Switch from random sampling to surrogate-guided threshold sampling.

    For the first *n_randomsamples* evaluations, points are drawn from a
    Sobol quasi-random sequence.  Once enough data has been collected the
    generator trains the provided surrogate and uses
    ``probabilistic_threshold_filter`` to bias sampling toward regions
    that are predicted to exceed a performance threshold.

    Parameters:
        domain:            ``InputDomain`` for the problem.
        surrogate:         Surrogate model (e.g. ``RandomForestEnsembleModel``).
        sampler:           QMC engine with a ``.random(n)`` method.
        n_randomsamples:   Number of evaluations before switching to threshold
                           sampling (default 512).
    """

    def __init__(self, domain, surrogate, sampler, n_randomsamples=512):
        super().__init__(domain)
        self.sampler = sampler
        self.surrogate = surrogate
        self.n_randomsamples = n_randomsamples

    def generate(self, state, n_candidates):
        if state.index is None or len(state.index) < self.n_randomsamples:
            # Pure random phase
            X_unit = self.sampler.random(n_candidates)
            Xs = self.domain.inverse_transform(X_unit)
        else:
            # Surrogate-guided threshold phase
            threshold_value = 0.1
            Xs, _, _, _ = probabilistic_threshold_filter(
                self.domain,
                state,
                self.sampler,
                self.surrogate,
                self.n_randomsamples,
                threshold_value,
            )
            Xs = Xs[:n_candidates]
        return Xs, None


# ── Configuration ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Total evaluations to perform
    total_evaluations = 2056
    # Number of cores available on this node
    NCPUS = 128

    # ── Paths (replace with your own) ─────────────────────────────────────
    base_paths = {
        "exe": EXECUTABLE,
        "input": "./input_base.nml",
        "inputs_dir": "./runs/inputs/",
        "run_dir": "./runs/",
    }
    io_params = {}

    # ── Resource manager ──────────────────────────────────────────────────
    resource_manager = ResourceManager(total_cores=NCPUS)

    # ── Simulator & scheduler ─────────────────────────────────────────────
    simulator = Simulator(base_paths, io_params)
    scheduler = MPIScheduler(output_dir="./runs/screen_outputs/")

    # ── Domain ────────────────────────────────────────────────────────────
    domain, X_names = InputDomain.read_json("./domain.json")

    # ── Empty initial state ───────────────────────────────────────────────
    state = State(
        input_domain=domain,
        index=None,
        Xs=None,
        Ys=None,
        X_names=X_names,
    )

    # ── Surrogate ─────────────────────────────────────────────────────────
    surrogate = SingleFidelityRandomForestSurrogate()

    # ── Candidate generator ───────────────────────────────────────────────
    generate_candidates = RandomThenThresholdCandidateGenerator(
        domain=domain,
        surrogate=surrogate,
        sampler=Sobol(d=domain.dim, scramble=True),
    )

    # ── Run ───────────────────────────────────────────────────────────────
    csv_name = "async_results.csv"

    run_async_loop(
        total_evaluations,
        generate_candidates,
        state,
        simulator,
        resource_manager,
        scheduler=scheduler,
        max_workers=8,
        csv_name=csv_name,
        poll_interval=10,
    )
