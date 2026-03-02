import os
import subprocess

from millefeuille.simulator import Scheduler

from Utils import safe_mkdir


def get_PBS_hosts(l_batch, nproc):
    if l_batch:
        # Reading PBS file
        import os

        machine_file = os.environ["PBS_NODEFILE"]
        with open(machine_file, "r") as machinefile:
            hosts = [line.rstrip() for line in machinefile]

        ntot_proc = len(hosts)

        if ntot_proc % nproc != 0:
            import sys

            print("ntot_proc and nproc incompatible, division remainder non zero...")
            print("Exiting...")
            sys.exit()

        safe_mkdir("./.tmp/")
        Nbatch = ntot_proc // nproc
    else:
        Nbatch = 1
        hosts = None

    return Nbatch, hosts


class PBSMPIScheduler(Scheduler):
    def __init__(self, output_dir: str, hosts: list[str], mpiexec_path: str = "mpiexec"):
        safe_mkdir(output_dir)
        self._output_dir = output_dir
        self._hosts = hosts
        self._mpiexec = mpiexec_path
        self._wdir = os.environ.get("PBS_O_WORKDIR", os.getcwd())

        safe_mkdir("./.tmp/")

    @property
    def mpiexec(self) -> str:
        return self._mpiexec

    @property
    def output_dir(self) -> str:
        return self._output_dir

    def launch_jobs(self, exe: str, nproc: int, inputs: list[str], indices: list[str]):
        """
        Launches MPI jobs across multiple hosts. Assumes input files are prepared.
        """
        Nbatch = len(inputs)
        Nhosts = len(set(self._hosts))
        jobs_per_host = Nbatch // Nhosts

        processes = []
        outs = []
        errs = []
        for ibatch, (index, input_path) in enumerate(zip(indices, inputs, strict=False)):
            output_file = os.path.join(self.output_dir, f"Simulator_std_output_{index}.txt")
            error_file = os.path.join(self.output_dir, f"Simulator_std_error_{index}.txt")

            out = open(output_file, "w")
            err = open(error_file, "w")

            host_file = f"./.tmp/hosts_batch_{ibatch}_{index}.txt"

            with open(host_file, "w") as f:
                for ihost in range(ibatch * nproc, (ibatch + 1) * nproc):
                    f.write(self._hosts[ihost] + "\n")

            if Nhosts == 1:
                exe_cmd = f'{self._mpiexec} -n {nproc} {exe} "{input_path}"'
            else:
                if jobs_per_host > 1:
                    iprocset = ibatch % jobs_per_host
                    exe_cmd = (
                        f"{self._mpiexec} -v6 -genv I_MPI_DEBUG=4 "
                        f"-host {self._hosts[ibatch * nproc]} "
                        f"-genv I_MPI_PIN_PROCESSOR_LIST={iprocset * nproc}-{(iprocset + 1) * nproc - 1} "
                        f'-wdir {self._wdir} -n {nproc} {exe} "{input_path}"'
                    )
                else:
                    host_list = ",".join(set(self._hosts[ibatch * nproc : (ibatch + 1) * nproc]))
                    exe_cmd = (
                        f"{self._mpiexec} -v6 -genv I_MPI_DEBUG=4 "
                        f'-host {host_list} -wdir {self._wdir} -n {nproc} {exe} "{input_path}"'
                    )

            print(f"Launching job {ibatch + 1}/{Nbatch}: {exe_cmd}")
            proc = subprocess.Popen(exe_cmd, stdout=out, stderr=err, shell=True)
            processes.append(proc)
            outs.append(out)
            errs.append(err)

        for proc, out, err in zip(processes, outs, errs, strict=False):
            proc.wait()
            out.close()
            err.close()
