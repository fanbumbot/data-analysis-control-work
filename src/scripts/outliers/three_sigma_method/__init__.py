from pandas import Series

from .results import get_three_sigma_results

def get_outliers_three_sigma(series: Series, normalize: bool):
    vars = get_three_sigma_results(series, normalize)
    outliers = series[(series < vars.left_bound) | (series > vars.right_bound)]
    return outliers, vars