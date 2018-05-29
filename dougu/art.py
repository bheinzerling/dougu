# based on:
# https://raw.githubusercontent.com/smartschat/art/master/art/significance_tests.py

from math import fabs
import numpy as np


def approximate_randomization_test(
        scores1, scores2, agg=np.average, trials=10000):
    """Compute the statistical significance of a difference between
    the syss via a paired two-sided approximate randomization test.

    Args:
        scores1: Scores of the first system under consideration.
        scores2: Scores of the second system under consideration.
        agg: Function which aggregates all scores for individual
                    documents to obtain a score for the whole corpus.
        trials: The number of iterations during the test. Defaults to 10000.
    Returns:
        An approximation of the probability of observing corpus-wide
        differences in scores at least as extreme as observed here, when
        there is no difference between the syss.
    """
    if not isinstance(scores1, np.ndarray):
        scores1 = np.array(scores1)
    if not isinstance(scores2, np.ndarray):
        scores2 = np.array(scores2)
    scores = np.stack([scores1, scores2]).T

    abs_diff = fabs(agg(scores[:, 0]) - agg(scores[:, 1]))
    shuffled_was_at_least_as_high = 0

    for i in range(0, trials):
        mask = np.random.rand(scores.shape[0]) >= 0.5
        mask = np.stack([mask, ~mask]).T
        pseudo_scores1 = scores[mask]
        pseudo_scores2 = scores[~mask]
        pseudo_diff = fabs(agg(pseudo_scores1) - agg(pseudo_scores2))
        if pseudo_diff >= abs_diff:
            shuffled_was_at_least_as_high += 1

    significance_level = (shuffled_was_at_least_as_high + 1) / (trials + 1)
    return significance_level


art = approximate_randomization_test


if __name__ == "__main__":
    n = 10000
    s1 = np.random.rand(n)
    s2 = np.random.rand(n)
    print(art(s1, s2))  # should be high
    print(art(s1, s2 + 0.1))  # should be low
