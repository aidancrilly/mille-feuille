"""
Generic executable simulator for the asynchronous scheduling example.

This is a dummy implementation showing the structure required to couple
an MPI-based simulator to mille-feuille's async loop.  Replace the
placeholder methods with your own I/O logic.

It makes the following assumptions about the simulator:

- The input deck is a Fortran namelist file; a template is copied and
  modified to create new input decks.

- Simulation inputs and outputs use the run index in the file name.

- A single fidelity is run, so each execution uses the same number of
  MPI ranks as specified in the input template.
"""

import fileinput
import shutil

import f90nml
import numpy as np
import pandas as pd
from millefeuille.simulator import ExectuableSimulator

from Utils import safe_mkdir


class Simulator(ExectuableSimulator):
    """Generic single-fidelity executable simulator.

    Parameters:
        base_paths (dict): Must contain keys ``"exe"``, ``"input"``,
            ``"run_dir"``, ``"inputs_dir"``.
        io_params (dict):  Optional overrides (currently unused — provided
            for forward compatibility with application-specific subclasses).
    """

    def __init__(self, base_paths, io_params):
        self.exe = base_paths["exe"]

        self.input_template = base_paths["input"]
        self.run_dir = base_paths["run_dir"]
        self.inputs_dir = base_paths["inputs_dir"]

        safe_mkdir(self.run_dir)
        safe_mkdir(self.inputs_dir)

        _, index_length = self.get_index_from_input(self.input_template)
        self.index_length = index_length

        self.nproc = self.get_nproc_from_dimsize(self.input_template)

    # ── Required properties ───────────────────────────────────────────────

    @property
    def executable(self):
        return self.exe

    @property
    def nproc_per_fidelity(self):
        return [self.nproc]

    # ── Prepare / launch / postprocess ────────────────────────────────────

    def single_prepare_inputs(self, index, x, s=None):
        """Copy the input template and inject run-specific parameters.

        Override the body of this method to write the actual parameter
        values from *x* into the namelist.
        """
        idx_str = str(index).zfill(self.index_length)
        input_file = f"{self.inputs_dir}input_{idx_str}.nml"
        shutil.copy(self.input_template, input_file)

        # NB: Insert your parameter values from x into the input file here.
        # This is a dummy implementation — replace with actual logic.
        input_dict = {
            "index": idx_str,
            "output_dir": f"'{self.run_dir}'",
        }

        self.replace_inputs(input_file, input_dict)

    def launch(self, indices, Xs, scheduler, Ss=None):
        """Invoke the scheduler to run the executable on prepared inputs."""
        input_paths = [f"{self.inputs_dir}input_{str(i).zfill(self.index_length)}.nml" for i in indices]
        str_indices = [str(i).zfill(self.index_length) for i in indices]
        scheduler.launch_jobs(self.exe, self.nproc, input_paths, str_indices)

    def single_postprocess(self, index, x, s=None):
        """Read simulation outputs and compute the objective value.

        Override this method with your own post-processing logic.
        """
        P = self._get_target_parameters(index)
        Y = self._compute_objective(P)
        return P, Y

    # ── Application-specific helpers (override these) ─────────────────────

    def _get_target_parameters(self, index):
        """Read diagnostic outputs for a given run index.

        Replace with your own file I/O.
        """
        idx_str = str(index).zfill(self.index_length)
        diag_file = f"{self.run_dir}/run_{idx_str}/diag_{idx_str}.csv"
        df = pd.read_csv(diag_file)

        # NB: Read your simulation diagnostics from df here.
        # This is a dummy implementation — replace with actual logic.
        Y_DT = df["Y_DT"].values[-1]

        return np.array([Y_DT])

    def _compute_objective(self, P):
        """Compute the scalar objective from the diagnostic vector *P*.

        Replace with your own cost / objective function.
        """
        return np.sum(P, keepdims=True)

    # ── Static I/O utilities ──────────────────────────────────────────────

    @staticmethod
    def get_nproc_from_dimsize(input_file):
        """Read MPI decomposition from a Fortran namelist template."""
        nml = f90nml.read(input_file)
        nproc = nml["grid"]["dimsize_0"] * nml["grid"]["dimsize_1"] * nml["grid"]["dimsize_2"]
        return nproc

    @staticmethod
    def get_index_from_input(input_file):
        """Read the run index and its zero-padded length from a namelist."""
        nml = f90nml.read(input_file)
        index = nml["problem"]["index"]
        if "index_length" in nml["problem"].keys():
            index_length = nml["problem"]["index_length"]
        else:
            index_length = len(str(index))
        return index, index_length

    @staticmethod
    def replace_inputs(input_file, parameter_dict):
        """Overwrite namelist values that match keys in *parameter_dict*."""
        for line in fileinput.input(input_file, inplace=True):
            for key, value in parameter_dict.items():
                if key.lower() == line.split("=")[0].strip().lower():
                    line = f"\t{key} = {value} \n"
            print(line, end="")
