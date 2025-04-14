from pandas import Series

from ...intermediary_vars import IntermediaryVariables

from .vars import get_iqr_vars

def get_outliers_iqr(series: Series, normalize: bool):
    vars = get_iqr_vars(series, normalize)
    outliers = series[(series < vars.left_bound) | (series > vars.right_bound)]
    return outliers, vars