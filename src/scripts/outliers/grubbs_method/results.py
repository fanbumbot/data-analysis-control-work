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

    vars += IntermediaryResults(
        t_max = abs(vars.max-vars.mean)/vars.std,
        t_min = abs(vars.min-vars.mean)/vars.std,
        t_crit = stats.t.ppf(q=1-alpha/2, df=series.count())
    )

    vars += IntermediaryResults(
        is_max_outlier = vars.t_max > vars.t_crit,
        is_min_outlier = vars.t_min > vars.t_crit
    )

    return vars
