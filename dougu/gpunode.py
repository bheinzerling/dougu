from pathlib import Path
import time
from contextlib import contextmanager
import random
import traceback
import json
from subprocess import run, PIPE
from copy import deepcopy
from uuid import uuid4

import paramiko
from boltons.iterutils import chunked

from dougu import random_string, args_to_str, mkdir
from dougu.results import Results


def get_gpu_node(
        sentinel, nodes_dir, proc_per_gpu, poll_period=300, log=None):
    while True:
        for proc in range(0, proc_per_gpu):
            try:
                for node in nodes_dir.iterdir():
                    for gpu in node.iterdir():
                        time.sleep(random.random())
                        if len(list(gpu.iterdir())) <= proc:
                            sentinel_file = gpu / sentinel
                            sentinel_file.touch(exist_ok=False)
                            if len(list(gpu.iterdir())) <= proc + 1:
                                return node.stem, gpu.stem, sentinel_file
                            sentinel_file.unlink()
                        else:
                            pass
            except:
                traceback.print_exc()

        (log.info if log else print)(
            f"no GPU available. Waiting {poll_period} seconds.")
        time.sleep(poll_period)


@contextmanager
def node_and_gpu(sentinel, nodes=Path("nodes"), proc_per_gpu=1, log=None):
    node, gpu, sentinel_file = get_gpu_node(
        sentinel, nodes, proc_per_gpu, log=log)
    (log.info if log else print)(f"{node} gpu:{gpu}")
    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(node, timeout=None)
    except:
        traceback.print_exc()
    try:
        yield ssh, gpu
    except:
        traceback.print_exc()
    finally:
        ssh.close()
        try:
            sentinel_file.unlink()
        except FileNotFoundError:
            pass


def run_on_node(
        args,
        *,
        pwd=".",
        env="pt5",
        file="main.py",
        nodes=Path("nodes"),
        log=None,
        positional_arg=None):
    """Run a Python file with args on one of the GPUs of one of
    the nodes listed in the nodes directory.
    """
    args.sentinel = random_string()
    info = log.info if log else print
    with node_and_gpu(
            args.sentinel,
            nodes=nodes,
            proc_per_gpu=args.proc_per_gpu,
            log=log) as (ssh, gpu):
        pre = f"cd {pwd} && . activate {env} &&"
        py = f"CUDA_VISIBLE_DEVICES={gpu} python {file}"
        args_str = args_to_str(args, positional_arg)
        command = " ".join((pre, py, args_str))
        info(command)
        stdin, stdout, stderr = ssh.exec_command(command, get_pty=True)
        for line in iter(stdout.readline, ""):
            info(line, end="")
        try:
            res_sentinel, json_str = line.strip().split("\t")
            assert res_sentinel == args.sentinel
            return json.loads(json_str)
        except:
            traceback.print_exc()
            info("args:", args)
            info("line:", line)
    # return run_on_node(args, pwd=pwd, env=env, file=file)


def slurm_script(
        args,
        *,
        job_name=None,
        outdir=Path("out/slurm").absolute(),
        done_dir=Path("out/slurm/done").absolute(),
        results_dir=Path("out/slurm/results").absolute(),
        output=None,
        error=None,
        partition="skylake-deep.p,pascal-deep.p,pascal-crunch.p",
        modules=[
            "CUDA/8.0.44",
            "OpenBLAS/0.2.13-GCC-4.8.4-LAPACK-3.5.0"],
        nodelist=None,
        exclude=None,
        time="8:00:00",
        cpus_per_task=1,
        n_gpus=1,
        pwd=".",
        env="pt5",
        file="main.py",
        log=None,
        positional_arg=None):
    """Write a SLURM script which will run the Python file with
    the supplied args.
    """
    for d in outdir, done_dir, results_dir:
        mkdir(d)
    batch_sentinel = random_string()
    pwd = Path(pwd).absolute()
    gres = f"gpu:{n_gpus}"
    output = output or (outdir / f"{batch_sentinel}.out")
    sbatch_args = {
        "job-name": job_name or pwd.name,
        "output": output,
        "error": error or (outdir / f"{batch_sentinel}.err"),
        "time": time,
        "ntasks": 1,
        "cpus-per-task": cpus_per_task,
        "gres": gres,
        "partition": partition}
    if nodelist:
        sbatch_args["nodelist"] = ",".join(nodelist)
    if exclude:
        sbatch_args["exclude"] = ",".join(exclude)

    py_cmds = []
    if not isinstance(args, list):
        args = [args]
    for a in args:
        if hasattr(a, "results_dir"):
            results_dir = a.results_dir.absolute()
        a.sentinel = batch_sentinel + "__" + random_string()
        a.result_file = results_dir / a.sentinel
        args_str = args_to_str(a, positional_arg)
        py_cmd = " ".join(["python", file, args_str])
        py_cmds.append(py_cmd)
    py = "\n\n".join(py_cmds)

    prelude = "\n".join(f"#SBATCH --{k}={v}" for k, v in sbatch_args.items())
    module_load = "\n".join(f"module load {m}" for m in modules)
    cmds = "\n".join([module_load, f"cd {pwd}", f". activate {env}", py])
    post_cmd = f"sleep 10\ncp {output} {done_dir}"
    script = "\n\n".join(["#!/bin/bash", prelude, cmds, "wait", post_cmd])

    script_file = outdir / f"{batch_sentinel}.sh"
    with script_file.open("w") as out:
        out.write(script)
    return script_file


def slurm_submit(args, **kwargs):
    script_file = slurm_script(args, **kwargs)
    sbatch_cmd = ["sbatch", script_file]
    sbatch_out = run(sbatch_cmd, encoding="utf8", stdout=PIPE).stdout
    msg = "Submitted batch job "
    assert sbatch_out[:len(msg)] == msg
    jobid = sbatch_out.strip().split()[-1]
    assert str(int(jobid)) == jobid
    return jobid


def get_jobs(args, configs, index, results):
    for config in configs:
        _args = deepcopy(args)
        _args.__dict__.update(config)
        _args.gpu_id = 0
        conf_key = tuple(getattr(_args, colname) for colname in index)
        try:
            n_done = results.n_done(conf_key)
        except:
            print(conf_key)
            raise
        for i in range(n_done, _args.trials_per_config):
            __args = deepcopy(_args)
            __args.runid = "__" + uuid4().hex
            __args.random_seed = i
            yield __args


def submit_and_collect(args, configs, index, columns, append_results_fn):
    """Create and submit SLURM jobs for each configuration in configs.
    Then collect results and store them."""
    total_configs = args.trials_per_config * len(configs)
    with Results(args.results_store, columns, index) as results:
        append_results_fn(args, results)
        jobs = list(get_jobs(args, configs, index, results))
        print("Total", total_configs, "configs.", "Todo:", len(jobs))
        if args.inspect_results:
            from IPython import embed
            embed()
            return
        if args.submit_jobs:
            random.shuffle(jobs)
            jobids = []
            for batch in chunked(jobs, args.configs_per_job):
                jobid = slurm_submit(
                    batch, positional_arg="command",
                    exclude=args.exclude, time=args.time,
                    job_name=args.job_name)
                print(jobid)
                time.sleep(1)
                jobids.append(jobid)
            print("Submitted", len(jobids), "jobs")
        while True:
            time.sleep(10)
            append_results_fn(args, results)
