from pathlib import Path
import time

import mlflow

from .log import WithLog
from .argparser import ConfigurableWithLog

import logging
logger = logging.getLogger('alembic.runtime.migration')
logger.disabled = True


class ExperimentLogger(Configurable, WithLog):
    args = [
        ('--backend-store-uri', dict(type=str, default='sqlite:///mlflow.db')),
        ('--mlflow-runid', dict(type=str)),
        ]

    def __init__(self, conf, exp_params):
        super().__init__(conf)
        self.exp_params = exp_params
        self.setup()

    def setup(self):
        self.exp_name = self.conf.exp_name
        self.log(
            f'mlflow backend for exp {self.exp_name}: '
            f'{self.conf.backend_store_uri}')
        mlflow.set_tracking_uri(self.conf.backend_store_uri)
        mlflow.set_experiment(self.conf.exp_name)
        self.exp = mlflow.get_experiment_by_name(self.conf.exp_name)

    def start_run(self):
        runid = self.conf.runid
        mlflow_runid = self.conf.mlflow_runid
        if mlflow_runid:
            mlflow.start_run(run_id=mlflow_runid)
            old_runid = mlflow.active_run().data.tags['mlflow.runName']
            assert old_runid == runid, (
                f'old run id {old_runid} != {runid}')
        else:
            mlflow.set_experiment(self.exp_name)
            mlflow.start_run(run_name=runid)
            self.log_params(self.exp_params)
        expid = self.exp.experiment_id
        mlflow_runid = mlflow.active_run().info.run_id
        self.log(f'mlflow runid:  {mlflow_runid}')
        f = 'file://'
        if self.exp.artifact_location.startswith(f):
            artifact_dir = Path(self.exp.artifact_location[len(f):])
            if not artifact_dir.exists():
                artifact_dir = Path('.').absolute() / 'mlflow' / expid
                artifact_uri = f + str(artifact_dir)
                mlflow.active_run().info._artifact_uri = artifact_uri

    def log_params(self, params):
        mlflow.log_params(params)

    def log_metrics(self, metrics, step=None):
        n_tries = 0
        while True:
            try:
                mlflow.log_metrics(metrics, step=step)
                break
            except Exception as e:
                if n_tries < 5:
                    n_tries += 1
                    time.sleep(10)
                else:
                    raise e

    def log_artifacts(self):
        for f in self.conf.rundir.iterdir():
            if f.suffix not in {'.pt', '.pth'}:
                try:
                    mlflow.log_artifact(f)
                except IOError:
                    print('failed to log artifact', f)

    def end_run(self):
        mlflow.end_run()
        self.log(f'Ended run: {self.conf.mlflow_runid}')

    def query_results(self):
        if not self.exp:
            return None
        params = dict(self.exp_params)
        params.pop('jobid')
        query = ' and '.join(f'param.{k} = "{v}"' for k, v in params.items())
        self.log(query)
        return mlflow.search_runs(self.exp.experiment_id, query)
