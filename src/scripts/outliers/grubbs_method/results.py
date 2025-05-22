from pandas import Series
import scipy.stats as stats

from ...intermediary_results import IntermediaryResults

def get_grubbs_results(series: Series, alpha: float):
    vars = IntermediaryResults(
        std = series.std(),
        mean = series.mean(),
        min = series.min(),
        max = series.max()
    )

    t_dist = stats.t.ppf(1-alpha / (2 * series.size), series.size - 2)
    t_crit = (series.size-1)/(series.size ** 0.5) * ((t_dist/(series.size - 2 + t_dist)) ** 0.5)

    vars += IntermediaryResults(
        t_max = abs(vars.max-vars.mean)/vars.std,
        t_min = abs(vars.min-vars.mean)/vars.std,
        t_crit = t_crit
    )

    vars += IntermediaryResults(
        is_max_outlier = bool(vars.t_max > vars.t_crit),
        is_min_outlier = bool(vars.t_min > vars.t_crit)
    )

    return vars
