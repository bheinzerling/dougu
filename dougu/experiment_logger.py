from pathlib import Path
import time
import json

import mlflow

from .log import WithLog
from .argparser import Configurable
from .misc import SubclassRegistry
from .decorators import cached_property
from .io import (
    mkdir,
    json_load,
    jsonlines_load,
    json_dump,
    ensure_serializable,
    )
from .iters import flatten

import logging
logger = logging.getLogger('alembic.runtime.migration')
logger.disabled = True


class ExperimentLogger(Configurable, SubclassRegistry, WithLog):
    args = [
        ('--exp-logger', dict(type=str, default='mlflowlogger')),
        ('--exp-name', dict(type=str, default='dev')),
        ('--exp-results-file', dict(type=Path, nargs='+')),
        ('--exp-results-ignore-param', dict(type=str, nargs='+')),
        ]

    def __init__(
            self,
            conf,
            exp_params,
            outdir=Path('out'),
            results_patch_data=None,
            ):
        super().__init__(conf)
        self.exp_params = exp_params
        self.outdir = outdir
        self.setup()
        self.results_patch_data = results_patch_data or {}

    def setup(self):
        raise NotImplementedError

    def start_run(self):
        self.log_params(self.exp_params)

    def end_run(self):
        pass

    def log_params(self, params):
        raise NotImplementedError

    def log_artifacts(self, params):
        raise NotImplementedError

    def log_metrics(self, metrics, step=None):
        raise NotImplementedError

    @property
    def results_dir(self):
        return mkdir(self.outdir / 'results' / self.conf.exp_name)


class FileLogger(ExperimentLogger):
    def setup(self):
        mkdir(self.log_dir)
        self.log(f'log dir: {self.log_dir}')

    @property
    def log_dir_prefix(self):
        return f'exp_{self.conf.exp_name}.log.'

    @property
    def log_dir_parent(self):
        try:
            return self.conf.rundir.parent
        except AttributeError:
            for dirname in ('out_dir', 'outdir'):
                try:
                    return getattr(self.conf, dirname)
                except:
                    pass
        raise ValueError()

    @property
    def log_dir(self):
        dirname = f'{self.log_dir_prefix}runid_' + self.conf.runid
        return self.log_dir_parent / dirname

    def log_params(self, params):
        params_file = self.log_dir / 'params.json'
        if params_file.exists():
            params = json_load(params_file) | params
        json_dump(ensure_serializable(params), params_file)

    def log_artifacts(self):
        pass

    def log_metrics(self, metrics, step=None):
        for metric, score in metrics.items():
            metric_file = self.log_dir / f'metric.{metric}.jsonl'
            with metric_file.open('a') as out:
                metric_dict = {'step': step, metric: score}
                out.write(json.dumps(metric_dict) + '\n')

        metrics_file = self.log_dir / 'metrics.jsonl'
        metrics = metrics | {'step': step}
        with metrics_file.open('a') as out:
            out.write(json.dumps(metrics) + '\n')

    @cached_property
    def results(self):
        return self._results(
            results_dir=self.results_dir,
            results_file=getattr(self.conf, 'exp_results_file', None),
            log=self.log,
            patch_data=self.results_patch_data,
            )

    @staticmethod
    def _results(
            *,
            results_dir=None,
            results_file=None,
            log=print,
            patch_data=None):
        import pandas as pd
        from tqdm import tqdm
        if results_file:
            results = list(flatten(map(jsonlines_load, results_file)))
        else:
            assert results_dir
            results = []
            for f in (tqdm(results_dir.iterdir())):
                try:
                    result = json_load(f)
                    results.append(result)
                except:
                    log(f'Exception while loading result from file\n{f}')

        # convert lists to tuples since tuples behave nicer in pandas
        results = [{
            k: tuple(v) if isinstance(v, list) else v
            for k, v in result.items()
            }
            for result in results
            ]
        df = pd.DataFrame(results)
        for key, value in (patch_data or {}).items():
            if key in df.columns:
                df[key].fillna(value, inplace=True)
            else:
                df[key] = value
        return df

    def remove_ignored_params(self, params):
        ignored = set(self.conf.exp_results_ignore_param or [])
        return {k: v for k, v in params.items() if k not in ignored}

    def results_with_params(self, params):
        import numpy as np
        params = ensure_serializable(params)
        params = self.remove_ignored_params(params)

        def equal(column, value):
            # https://github.com/pandas-dev/pandas/issues/20442
            # pandas treats None values in object columns as np.nan,
            # which is not equal to None
            if value is None:
                return column.isna()
            # pandas stores lists as tuples, so we have to convert
            # lists to tuple to make equality checking work as intended
            if isinstance(value, list):
                value = tuple(value)
            return column == value

        df = self.results
        if df.empty:
            return df
        column_masks = [equal(df[k], v) for k, v in params.items()]
        mask = np.logical_and.reduce(column_masks)
        return df[mask]

    def is_done(self, params, metrics=None):
        import numpy as np
        results = self.results_with_params(params)
        if metrics is not None:
            column_masks = [~results[metric].isna() for metric in metrics]
            metrics_mask = np.logical_and.reduce(column_masks)
            results = results[metrics_mask]
        return not results.empty


class MlflowLogger(ExperimentLogger):
    args = [
        ('--backend-store-uri', dict(type=str, default='sqlite:///mlflow.db')),
        ('--mlflow-runid', dict(type=str)),
        ]

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

    @property
    def results(self):
        if not self.exp:
            return None
        params = dict(self.exp_params)
        try:
            params.pop('jobid')
        except KeyError:
            pass
        query = ' and '.join(f'param.{k} = "{v}"' for k, v in params.items())
        self.log(query)
        return mlflow.search_runs(self.exp.experiment_id, query)
