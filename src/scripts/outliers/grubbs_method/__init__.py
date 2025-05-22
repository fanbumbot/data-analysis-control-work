from pandas import Series

from .results import get_grubbs_results

from ...intermediary_results import IntermediaryResults

def get_outliers_grubbs(series: Series, alpha: float):
    all_vars = list()

    series_without_outliers = list(series)
    outliers = list()

    while True:
        vars = get_grubbs_results(Series(series_without_outliers), alpha)
        all_vars.append(vars)
        
        if vars.is_max_outlier and vars.is_min_outlier:
            series_without_outliers.remove(vars.max)
            series_without_outliers.remove(vars.min)
            outliers.append(vars.max)
            outliers.append(vars.min)
        elif vars.is_max_outlier:
            series_without_outliers.remove(vars.max)
            outliers.append(vars.max)
        elif vars.is_min_outlier:
            series_without_outliers.remove(vars.min)
            outliers.append(vars.min)
        else:
            break

    return outliers, all_vars, Series(series_without_outliers)
