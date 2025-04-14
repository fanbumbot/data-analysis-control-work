from pandas import Series

from .vars import get_grubbs_vars

from ...intermediary_vars import IntermediaryVariables

def get_outliers_grubbs(series: Series, alpha: float):
    all_vars = list()

    series_without_outliers = series.copy()
    outliers = list()

    while True:
        vars = get_grubbs_vars(series_without_outliers, alpha)
        all_vars.append(vars)
        
        if vars.is_max_outliers and vars.is_min_outliers:
            series_without_outliers = series_without_outliers.drop(vars.max, vars.min)
            outliers.append(vars.max)
            outliers.append(vars.min)
        elif vars.is_max_outliers:
            series_without_outliers = series_without_outliers.drop(vars.max)
            outliers.append(vars.max)
        elif vars.is_min_outliers:
            series_without_outliers = series_without_outliers.drop(vars.min)
            outliers.append(vars.min)
        else:
            break

    return outliers, all_vars, series_without_outliers
