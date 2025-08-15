import os
import subprocess
from typing import Sequence

from millefeuille.simulator import Scheduler


class ShellScheduler(Scheduler):
    def _is_sequence(self, seq):
        return type(seq) in [list, tuple]

    @property
    def mpiexec(self) -> str:
        """Path to `mpiexec` or equivalent MPI launch command."""
        return "mpiexec"

    @property
    def output_dir(self) -> str:
        """Directory to store stdout/stderr log files."""
        return "."

    def launch_jobs(self, exe: str, nprocs: Sequence[int], inputs: Sequence, indices: Sequence[str]):
        """
        Launches simulation jobs in parallel.

        Parameters:
            exe: Path to the MPI-enabled executable
            nprocs: Number of processes per job
            inputs: List of input file paths
            indices: List of run indices (for logging/output naming)
        """
        Nbatch = len(inputs)

        processes = []
        outs = []
        errs = []
        for ibatch, (index, nproc, job_inputs) in enumerate(zip(indices, nprocs, inputs, strict=False)):
            output_file = os.path.join(self.output_dir, f"std_output_{index}.txt")
            error_file = os.path.join(self.output_dir, f"std_error_{index}.txt")

            out = open(output_file, "w")
            err = open(error_file, "w")

            exe_cmd = f"{self.mpiexec} -n {nproc} -wdir {self.output_dir} {exe}"
            cmd_line_args = " "
            if job_inputs is not None:
                if self._is_sequence(job_inputs):
                    for job_input in job_inputs:
                        cmd_line_args += f'"{job_input}" '
                else:
                    cmd_line_args = f'"{job_inputs}"'
            exe_cmd += cmd_line_args

            print(f"Launching job {ibatch + 1}/{Nbatch}: {exe_cmd}")
            proc = subprocess.Popen(exe_cmd, stdout=out, stderr=err, shell=True)
            processes.append(proc)
            outs.append(out)
            errs.append(err)

        for proc, out, err in zip(processes, outs, errs, strict=False):
            proc.wait()
            out.close()
            err.close()

    def cleanup(self, indices):
        for index in indices:
            output_file = os.path.join(self.output_dir, f"std_output_{index}.txt")
            error_file = os.path.join(self.output_dir, f"std_error_{index}.txt")
            os.remove(output_file)
            os.remove(error_file)
