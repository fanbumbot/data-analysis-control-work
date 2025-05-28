from pandas import Series

from .grubbs_method import get_grubbs_results
from .iqr_method import get_iqr_results
from .three_sigma_method import get_three_sigma_results

def remove_outliers(series: Series, alpha: float):
    data = series.copy().astype(float)

    last_replacements = 0
    all_replacements = 0

    value_for_replace = data.median()

    while True:
        """
        grubbs_results = get_grubbs_results(data, alpha)

        is_max_outlier = grubbs_results.is_max_outlier
        is_min_outlier = grubbs_results.is_min_outlier

        idx_min = data.idxmin()
        idx_max = data.idxmax()

        if is_max_outlier and is_min_outlier:
            data.loc[idx_max] = value_for_replace
            data.loc[idx_min] = value_for_replace
            all_replacements += 2
        elif is_max_outlier:
            data.loc[idx_max] = value_for_replace
            all_replacements += 1
        elif is_min_outlier:
            data.loc[idx_min] = value_for_replace
            all_replacements += 1
        """
        """        
        iqr_results = get_iqr_results(data, True)
        outliers = data[(data < iqr_results.left_bound) | (data > iqr_results.right_bound)]
        all_replacements += len(outliers)
        data[(data < iqr_results.left_bound) | (data > iqr_results.right_bound)] = value_for_replace
        """
        
        
        three_sigma_results = get_three_sigma_results(data, True)
        outliers = data[(data < three_sigma_results.left_bound) | (data > three_sigma_results.right_bound)]
        all_replacements += len(outliers)
        data[(data < three_sigma_results.left_bound) | (data > three_sigma_results.right_bound)] = value_for_replace
        
        if last_replacements == all_replacements:
            break

        last_replacements = all_replacements

    return data, all_replacements