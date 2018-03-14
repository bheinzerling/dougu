from functools import wraps

from hyperopt import fmin, tpe, Trials, space_eval
from hyperopt import STATUS_OK, STATUS_FAIL  # NOQA
import joblib

from .io import json_dump


def hyperoptimize(rundir, space, ntrials=100, algo=tpe.suggest):
    """Peform hyperparameter optimization using the hyperopt package.
    The objective function takes hparams and returns a loss.
    The wrapper saves results by dumping the Trials object."""
    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            try:
                trials = joblib.load(rundir / "trials.pkl")
                joblib.dump(trials, rundir / "trials.pkl.bak")
            except:
                trials = Trials()
            for trial in range(len(trials), ntrials):
                best = fmin(
                    function,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=trial + 1,
                    trials=trials)

                joblib.dump(trials, rundir / "trials.pkl")
                best_hparams = space_eval(space, best)
                print(trial, "best", best_hparams)
                json_dump(best_hparams, rundir / "best_hparams.json")
        return wrapper
    return decorator

