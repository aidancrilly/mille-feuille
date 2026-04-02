"""
Single-node MPI scheduler for the asynchronous scheduling example.

NOTE: This scheduler assumes all jobs run on a **single node**.
      For multi-node PBS/Slurm scheduling see the ``loops/`` example.
"""

import os
import subprocess
import threading

from millefeuille.simulator import Scheduler

# local utils
from Utils import safe_mkdir


class MPIScheduler(Scheduler):
    """Launch MPI jobs on a single node via ``mpiexec``.

    Each call to ``launch_jobs`` blocks until all submitted processes
    finish.  Because the async loop dispatches work from separate
    threads, multiple jobs can be in flight concurrently.

    Parameters:
        output_dir:   Directory for stdout / stderr log files.
        mpiexec_path: Path to the ``mpiexec`` binary (default: ``"mpiexec"``).
    """

    def __init__(self, output_dir: str, mpiexec_path: str = "mpiexec"):
        safe_mkdir(output_dir)
        self._output_dir = output_dir
        self._mpiexec = mpiexec_path

    @property
    def mpiexec(self) -> str:
        return self._mpiexec

    @property
    def output_dir(self) -> str:
        return self._output_dir

    def launch_jobs(self, exe: str, nproc: int, inputs: list[str], indices: list[str]):
        """Launch MPI jobs sequentially on the local node.

        For each (index, input_path) pair a new ``mpiexec -n <nproc>``
        process is spawned.  The method blocks until every process has
        finished.

        Parameters:
            exe:     Path to the MPI-enabled executable.
            nproc:   Number of MPI ranks per job.
            inputs:  List of input file paths (one per job).
            indices: List of run index strings (used in log file names).
        """
        processes = []
        file_handles = []

        for index, input_path in zip(indices, inputs):
            output_file = os.path.join(self.output_dir, f"Simulator_output_{index}.txt")
            error_file = os.path.join(self.output_dir, f"Simulator_error_{index}.txt")

            out = open(output_file, "w")
            err = open(error_file, "w")
            file_handles.append((out, err))

            # Single-node launch: no host files needed
            exe_cmd = [self._mpiexec, "-n", str(nproc), exe] + input_path.split()

            print(f"[{threading.current_thread().name}] Launching: {' '.join(exe_cmd)}")
            proc = subprocess.Popen(exe_cmd, stdout=out, stderr=err)
            processes.append(proc)

        # Wait for all processes in this batch to complete
        for proc, (out, err) in zip(processes, file_handles):
            rc = proc.wait()
            print(f"[{threading.current_thread().name}] pid={proc.pid} finished rc={rc}")
            out.close()
            err.close()
