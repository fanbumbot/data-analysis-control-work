from pandas import Series

from ...intermediary_results import IntermediaryResults

from .results import get_iqr_results

def get_outliers_iqr(series: Series, normalize: bool):
    vars = get_iqr_results(series, normalize)
    outliers = series[(series < vars.left_bound) | (series > vars.right_bound)]
    return outliers, vars