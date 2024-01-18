from pathlib import Path
import time
import json
from numbers import Number

import numpy as np

from .log import WithLog
from .argparser import Configurable
from .misc import SubclassRegistry
from .decorators import cached_property
from .io import (
    mkdir,
    json_load,
    json_dump,
    jsonlines_load,
    jsonlines_dump,
    ensure_serializable,
    )
from .iters import flatten

import logging
logger = logging.getLogger('alembic.runtime.migration')
logger.disabled = True


class ExperimentLogger(Configurable, SubclassRegistry, WithLog):
    args = [
        ('--exp-logger', dict(type=str, default='filelogger')),
        ('--exp-logger-no-log', dict(action='store_true')),
        ('--exp-name', dict(type=str, default='dev')),
        ('--exp-results-file', dict(type=Path, nargs='+')),
        ('--exp-results-ignore-param', dict(type=str, nargs='+')),
        ]

    def __init__(
            self,
            conf,
            *,
            outdir=Path('out'),
            results_patch_data=None,
            ):
        super().__init__(conf)
        self.outdir = outdir
        if not conf.exp_logger_no_log:
            self.setup()
        self.results_patch_data = results_patch_data or {}

    def setup(self):
        raise NotImplementedError

    def start_run(self, exp_params):
        self.exp_params = exp_params
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

    @property
    def results_file(self):
        fname = f'{self.conf.exp_name}.jsonl'
        results_file = mkdir(self.outdir / 'results') / fname
        if not results_file.exists():
            self.cache_results(results_file)
        assert results_file.exists()
        return results_file

    def cache_results(self, cache_file):
        results = self.collect_results(self.results_dir, log=self.log)
        jsonlines_dump(results, cache_file)
        self.log(f'{len(results)} results written to {cache_file}')


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
                except Exception:
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
            recollect_results=getattr(self.conf, 'recollect_results', False),
            )

    @staticmethod
    def collect_results(results_dir, log=print):
        assert results_dir
        results = []
        from tqdm import tqdm
        for f in tqdm(results_dir.iterdir()):
            try:
                result = json_load(f)
                results.append(result)
            except Exception:
                log(f'Exception while loading result from file\n{f}')
        return results

    def _results(
            self,
            *,
            results_dir=None,
            results_file=None,
            log=print,
            patch_data=None,
            recollect_results=False,
            ):
        import pandas as pd
        if recollect_results:
            results = FileLogger.collect_results(results_dir, log=log)
            self.cache_results(self.results_file)
        else:
            results_file = results_file or [self.results_file]
            if results_file:
                results = list(flatten(map(jsonlines_load, results_file)))
                log(f'loaded {len(results)} results from {results_file}')
                recollect_results = False
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
                df[key].fillna(value or '', inplace=True)
            else:
                df[key] = value
        return df

    def remove_ignored_params(self, params):
        ignored = set(self.conf.exp_results_ignore_param or [])
        return {k: v for k, v in params.items() if k not in ignored}

    def results_with_params(self, params):
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

        num_params = {k: v for k, v in params.items() if isinstance(v, Number)}
        if num_params:
            num_query = ' and '.join(f'{k}=={v}' for k, v in num_params.items())
            df = df.query(num_query)
            params = dict(params.items() - num_params.items())

        none_params = {k: v for k, v in params.items() if v is None}
        if none_params:
            column_masks = [equal(df[k], v) for k, v in none_params.items()]
            mask = np.logical_and.reduce(column_masks)
            df = df[mask]
            params = dict(params.items() - none_params.items())

        string_params = {k: v for k, v in params.items() if isinstance(v, str)}
        params = dict(params.items() - string_params.items())

        if params:
            column_masks = [equal(df[k], v) for k, v in params.items()]
            mask = np.logical_and.reduce(column_masks)
            df = df[mask]

        # string comparisons seem to be slowest, so do them last when the dataframe
        # has been filtered down the most
        if string_params:
            string_query = ' and '.join(f'{k}=="{v}"' for k, v in string_params.items())
            df = df.query(string_query)

        # mask = None
        # for k, v in params.items():
        #     column_mask = equal(df[k], v)
        #     if mask is None:
        #         mask = column_mask
        #     else:
        #         mask &= column_mask
        #     if not mask.any():
        #         break
        return df

    def is_done(self, params, metrics=None):
        results = self.results_with_params(params)
        if metrics is not None:
            column_masks = [~results[metric].isna() for metric in metrics]
            metrics_mask = np.logical_and.reduce(column_masks)
            results = results[metrics_mask]
        return not results.empty

    def missing_params(self, params):
        missing = {}
        for k, v in params.items():
            if not self.is_done({k: v}):
                missing[k] = v
        return missing


class MlflowLogger(ExperimentLogger):
    args = [
        ('--backend-store-uri', dict(type=str, default='sqlite:///mlflow.db')),
        ('--mlflow-runid', dict(type=str)),
        ]

    @cached_property
    def mlflow(self):
        import mlflow
        return mlflow


    def setup(self):
        self.exp_name = self.conf.exp_name
        self.log(
            f'mlflow backend for exp {self.exp_name}: '
            f'{self.conf.backend_store_uri}')
        self.mlflow.set_tracking_uri(self.conf.backend_store_uri)
        self.mlflow.set_experiment(self.conf.exp_name)
        self.exp = mlflow.get_experiment_by_name(self.conf.exp_name)

    def start_run(self):
        runid = self.conf.runid
        mlflow_runid = self.conf.mlflow_runid
        if mlflow_runid:
            mlflow.start_run(run_id=mlflow_runid)
            old_runid = self.mlflow.active_run().data.tags['mlflow.runName']
            assert old_runid == runid, (
                f'old run id {old_runid} != {runid}')
        else:
            self.mlflow.set_experiment(self.exp_name)
            self.mlflow.start_run(run_name=runid)
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
        self.mlflow.log_params(params)

    def log_metrics(self, metrics, step=None):
        n_tries = 0
        while True:
            try:
                self.mlflow.log_metrics(metrics, step=step)
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
                    self.mlflow.log_artifact(f)
                except IOError:
                    print('failed to log artifact', f)

    def end_run(self):
        self.mlflow.end_run()
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
        return self.mlflow.search_runs(self.exp.experiment_id, query)
