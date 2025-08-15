import fileinput
import shutil

import f90nml
import numpy as np
import pandas as pd
from millefeuille.simulator import ExectuableSimulator

from Utils import safe_mkdir


class Simulator(ExectuableSimulator):
    """
    A dummy implementation to show the logic of mille-feuille coupling to a simulator

    It makes the following assumptions about the simulator:

    - The input deck is a fortran namelist file, a template is provided at initialisation
    that is copied and modified to create new input decks

    - Simulation inputs and outputs use the run index in the file name as marker

    - All key simulation parameters are written to a summary csv to be read by pandas

    """

    def __init__(self, base_paths, io_params):
        self.exe = base_paths["exe"]

        self.input_template = base_paths["input"]

        self.inputs_dir = base_paths["inputs_dir"]

        safe_mkdir(self.run_dir)
        safe_mkdir(self.inputs_dir)

        _, index_length = self.get_index_from_input(self.input_template)

        self.index_length = index_length

        self.nproc = self.get_nproc_from_dimsize(self.input_template)

    @property
    def executable(self):
        return self.exe

    @property
    def nproc_per_fidelity(self):
        return [self.nproc]

    def single_prepare_inputs(self, index, x, s=None):
        input_file = f"{self.inputs_dir}input_{str(index).zfill(self.index_length)}.nml"
        shutil.copy(self.input_template, input_file)

        # NB NEED TO INSERT X VALUES INTO INPUT FILE
        # This is a dummy implementation, replace with actual logic
        input_dict = {
            "index": str(index).zfill(self.index_length),
            "output_dir": f"'{self.run_dir}'",
        }

        self.replace_inputs(input_file, input_dict)

    def launch(self, indices, Xs, scheduler, Ss=None):
        input_paths = [f"{self.inputs_dir}/input_{str(i).zfill(self.index_length)}.f90" for i in indices]
        str_indices = [str(i).zfill(self.index_length) for i in indices]
        scheduler.launch_jobs(self.exe, self.nproc, input_paths, str_indices)

    def single_postprocess(self, index, x, s=None):
        P = self._get_target_parameters(index)
        Y = self._compute_objective(P)
        return P, Y

    def _get_target_parameters(self, index):
        # This grabs values from a diagnostic file produced by simulation
        # For example, csv dumped in run directories
        index_str = str(index).zfill(self.index_length)
        diag_file = f"{self.run_dir}/run_{index_str}/diag_{index_str}.csv"
        df = pd.read_csv(diag_file)

        # NB NEED TO READ APPROPRIATE P VALUES FOR df
        # This is a dummy implementation, replace with actual logic
        Y_DT = df["Y_DT"].values[-1]

        return np.array([Y_DT])

    def _compute_objective(self, P):
        # Use Ps to compute objective function Y
        # This is a dummy implementation, replace with actual logic

        return np.sum(P, keepdims=True)

    @staticmethod
    def get_nproc_from_dimsize(input_file):
        nml = f90nml.read(input_file)
        nproc = nml["grid"]["dimsize_0"] * nml["grid"]["dimsize_1"] * nml["grid"]["dimsize_2"]
        return nproc

    @staticmethod
    def get_index_from_input(input_file):
        nml = f90nml.read(input_file)
        index = nml["problem"]["index"]
        if "index_length" in nml["problem"].keys():
            index_length = nml["problem"]["index_length"]
        else:
            index_length = len(str(index))
        return index, index_length

    @staticmethod
    def replace_inputs(input_file, parameter_dict):
        for line in fileinput.input(input_file, inplace=True):
            for key, value in parameter_dict.items():
                # Check that key is in line and isn't a substring
                # lower() makes sure is case insensitive
                if key.lower() == line.split("=")[0].strip().lower():
                    line = f"\t{key} = {value} \n"
            print(line, end="")
