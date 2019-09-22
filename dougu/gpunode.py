from pathlib import Path
import time
from contextlib import contextmanager
import random
import traceback
import json
from subprocess import run, PIPE
from copy import deepcopy
from uuid import uuid4
import os

from boltons.iterutils import chunked

from dougu import random_string, args_to_str, mkdir
from dougu.results import Results

import warnings
import pandas.errors


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
    import paramiko
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
            "CUDA/8.0.61_375.26-GCC-5.4.0-2.26"
            ],
        nodelist=None,
        exclude=None,
        time="8:00:00",
        cpus_per_task=None,
        ntasks_per_node=None,
        n_gpus=1,
        pwd=".",
        env="pt1",
        file="main.py",
        log=None,
        positional_arg=None,
        verbose=False):
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
        "cpus-per-task": cpus_per_task or 1,
        "gres": gres,
        "partition": partition}
    if nodelist:
        sbatch_args["nodelist"] = ",".join(nodelist)
    if exclude:
        sbatch_args["exclude"] = ",".join(exclude)
    if ntasks_per_node:
        sbatch_args["ntasks-per-node"] = ntasks_per_node

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
    setup_cmds = """
export __GL_SHADER_DISK_CACHE="0"
"""
    cmds = "\n".join([module_load, f"cd {pwd}", f". activate {env}", py])
    post_cmd = f"sleep 10\ncp {output} {done_dir}"
    script = "\n\n".join([
        "#!/bin/bash", prelude, setup_cmds, cmds, "wait", post_cmd])
    if verbose:
        print(script)
    script_file = outdir / f"{batch_sentinel}.sh"
    with script_file.open("w") as out:
        out.write(script)
    return script_file


def slurm_submit(conf, **kwargs):
    script_file = slurm_script(conf, **kwargs)
    sbatch_cmd = ["sbatch", script_file]
    sbatch_out = run(sbatch_cmd, encoding="utf8", stdout=PIPE).stdout
    msg = "Submitted batch job "
    assert sbatch_out[:len(msg)] == msg
    jobid = sbatch_out.strip().split()[-1]
    assert str(int(jobid)) == jobid
    return jobid


def grid_engine_script(
        conf,
        pwd=Path("."),
        env="pt1",
        file="main.py",
        outdir=None,
        log=None,
        positional_arg=None,
        verbose=False,
        jobname=None,
        stdout_file='$HOME/uge/$JOB_ID'):

    if not isinstance(conf, list):
        confs = [conf]
    else:
        confs = conf

    preamble = [
        '#! /usr/bin/env bash',
        '-cwd',
        '-j y',
        '-o ' + stdout_file]
    if jobname:
        preamble.append('-N ' + jobname)
    if hasattr(conf[0], 'ac') and getattr(conf[0], 'ac'):
        preamble.append('-ac ' + conf[0].ac)
    if hasattr(conf[0], 'jc') and getattr(conf[0], 'jc'):
        preamble.append('-jc ' + conf[0].jc)
    preamble = '\n#$ '.join(preamble) + '\n'

    py_cmds = [
        " ".join(["python", file, args_to_str(conf, positional_arg)])
        for conf in confs]

    home = os.environ['HOME']
    cd_cmd = f"cd {pwd.absolute()}"
    env_cmd = f""". {home}/conda/bin/activate {env} && if [ -f $HOME/netvars.sh ]; then . $HOME/netvars.sh; fi"""  # NOQA
    if conf[0].cycle_gpus > 0:
        import shlex
        py = [
            ' && '.join([cd_cmd, env_cmd, shlex.quote(py_cmd)])
            for py_cmd in py_cmds]
        script = preamble + '\n'.join([
            f'xargs -P{conf[0].cycle_gpus} -I% bash -c "%" <<EOF',
            *py,
            'EOF'])
    else:
        py = "\n\n".join(py_cmds)
        script = "\n".join([preamble, cd_cmd, env_cmd, py])
    if verbose:
        print(script)
    if outdir is None:
        outdir = Path(home) / 'uge'
    script_file = mkdir(outdir.absolute()) / f"{conf[0].runid}.sh"
    with script_file.open("w") as out:
        out.write(script)
    return script_file


def grid_engine_submit(conf, positional_arg):
    script_file = grid_engine_script(conf, positional_arg=positional_arg)
    print(script_file)
    jobid = 1
    submit_args = {
        'group': ['-g', lambda: f'{conf[0].group}'],
        'queue': ['-l', lambda: f'{conf[0].queue}=1'],
        'time': ['-l', lambda: f'h_rt={conf[0].time}'],
        }
    submit_cmd = ['qsub', '-j', 'y', '-cwd']
    for arg, params in submit_args.items():
        if hasattr(conf[0], arg) and getattr(conf[0], arg):
            submit_cmd.extend((params[0], params[1]()))
    submit_cmd.append(script_file)
    submit_out = run(submit_cmd, encoding="utf8", stdout=PIPE).stdout
    print(submit_out)
    jobid = submit_out.split()[2]
    assert str(int(jobid)) == jobid
    return int(jobid)


def get_jobs(args, configs, index, results):
    warnings.simplefilter('ignore', pandas.errors.PerformanceWarning)
    try:
        df = results.dataframe
    except KeyError:
        df = None
    for config in configs:
        _args = deepcopy(args)
        _args.__dict__.update(config)
        # _args.gpu_id = 0
        conf_key = tuple(getattr(_args, colname) for colname in index)
        try:
            if df is None:
                raise KeyError
            n_done = len(df.loc(axis=0)[conf_key])
        except (KeyError, TypeError):
            n_done = 0
        for i in range(n_done, _args.trials_per_config):
            __args = deepcopy(_args)
            __args.runid = "__" + uuid4().hex
            if hasattr(args, "random_random_seed") and args.random_random_seed:
                __args.random_seed = random.randint(0, 2**30)
            else:
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
        if hasattr(args, "print_configs") and args.print_configs:
            for job in jobs:
                print(job)
        if args.submit_jobs:
            random.shuffle(jobs)
            jobids = []
            try:
                n_gpus = args.n_gpus
            except AttributeError:
                n_gpus = 1
            # submit = {
            #     'slurm': slurm_submit,
            #     'grid_engine': grid_engine_submit
            #     }[args.cluster_scheduler]
            positional_arg = "command"
            for batch in chunked(jobs, args.configs_per_job):
                if args.cluster_scheduler == 'slurm':
                    jobid = slurm_submit(
                        batch,
                        positional_arg=positional_arg,
                        partition=args.partition,
                        exclude=args.exclude,
                        ntasks_per_node=args.ntasks_per_node,
                        cpus_per_task=args.cpus_per_task,
                        n_gpus=n_gpus,
                        time=args.time,
                        job_name=args.job_name)
                elif args.cluster_scheduler == 'grid_engine':
                    jobid = grid_engine_submit(
                        batch, positional_arg=positional_arg)
                print(jobid)
                time.sleep(1)
                jobids.append(jobid)
            print("Submitted", len(jobids), "jobs")
        if args.collect_jobs:
            while True:
                time.sleep(10)
                append_results_fn(args, results)
