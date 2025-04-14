from pandas import Series
import scipy.stats as stats

from ...intermediary_vars import IntermediaryVariables

def get_grubbs_vars(series: Series, alpha: float):
    vars = IntermediaryVariables(
        std = series.std(),
        mean = series.mean(),
        min = series.min(),
        max = series.max()
    )

    vars += IntermediaryVariables(
        t_max = abs(vars.max-vars.mean)/vars.std,
        t_min = abs(vars.min-vars.mean)/vars.std,
        t_crit = stats.t.ppf(q=1-alpha/2, df=series.count())
    )

    vars += IntermediaryVariables(
        is_max_outlier = vars.t_max > vars.t_crit,
        is_min_outlier = vars.t_min > vars.t_crit
    )

    return vars
