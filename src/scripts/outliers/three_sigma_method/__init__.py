from pandas import Series

from .vars import get_three_sigma_vars

def get_outliers_three_sigma(series: Series, normalize: bool):
    vars = get_three_sigma_vars(series, normalize)
    outliers = series[(series < vars.left_bound) | (series > vars.right_bound)]
    return outliers, vars