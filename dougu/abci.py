import os
from pathlib import Path

from .iters import is_non_string_iterable


def submission_script(
        *,
        jobclass,
        jobname,
        cmd,
        max_job_duration='12:00:00',
        work_dir=os.getcwd(),
        py_env_name=os.environ['py_env_name'],
        stdout_dir=os.environ['uge_stdout_dir'],
        conda_activate='~/conda/bin/activate'
        ):

    assert py_env_name

    if is_non_string_iterable(cmd):
        cmd = '\n'.join(cmd)

    # write a batch script to /tmp
    # script=$(mktemp -u -p ~/submit_scripts).sh
    # echo Submitting job $jobname via batch script $script >&2
    # cat > $script <<EOF
    script = f"""#! /usr/bin/env bash
#$ -l {jobclass}
#$ -l h_rt={max_job_duration}
#$ -j y
#$ -o {stdout_dir}/$JOB_ID
#$ -N {jobname}

# activate virtual environment
. {conda_activate} {py_env_name}

cd {work_dir}

# the main command to run
{cmd}
"""
    return script


def submit_commands(
        cmds,
        *,
        jobclass,
        jobname,
        cmd_prefix='python main.py ',
        max_jobs=100,
        cmds_per_job=None,
        group=os.environ['group'],
        script_dir=Path('~/uge').expanduser(),
        **submission_kwargs,
        ):
    from tempfile import NamedTemporaryFile
    from subprocess import run, PIPE, STDOUT
    import shutil
    from boltons.iterutils import chunked

    if not cmds:
        return

    assert group

    cmds = [cmd_prefix + cmd for cmd in cmds]
    if cmds_per_job is None:
        import math
        cmds_per_job = math.ceil(len(cmds) / max_jobs)

    for cmds_batch in chunked(cmds, cmds_per_job):
        script = submission_script(
            jobclass=jobclass,
            jobname=jobname,
            cmd=cmds_batch,
            **submission_kwargs,
            )
        with NamedTemporaryFile(mode='w', delete=False) as tmp_script_file:
            tmp_script_file.write(script)
        submit_cmd = ['qsub', '-g', group, tmp_script_file.name]
        out = run(
            submit_cmd,
            encoding="utf8",
            stdout=PIPE,
            stderr=STDOUT,
            ).stdout
        jobid = int(out.split()[2])
        script_file = script_dir / f'{jobid}.sh'
        shutil.copy(tmp_script_file.name, str(script_file))
        print(script_file)
