from pandas import Series

from ..intermediary_results import IntermediaryResults

import math

def get_medians_method_results(series: Series):
    median = series.median()

    vars = IntermediaryResults(
        size = series.size,
        median = median
    )

    vars += IntermediaryResults(
        a_crit = math.floor((vars.size + 1)/2 - 1.96 * ((vars.size - 1) ** 0.5)),
        b_crit = math.floor(3.3 * math.log(vars.size) + 1)
    )

    a = 0
    b = 0

    current_series_size = 0

    is_greater_than_median_arr = list()

    is_last_greater_than_median = None
    for value in series:
        if value == median:
            is_greater_than_median_arr.append(None)
            continue
        is_greater_than_median = value > median
        is_greater_than_median_arr.append(is_greater_than_median)

        if ((is_last_greater_than_median == None) or
            (is_last_greater_than_median and not is_greater_than_median) or
            (not is_last_greater_than_median and is_greater_than_median)
        ):
            a += 1
            b = max(b, current_series_size)
            current_series_size = 0

    vars += IntermediaryResults(
        is_greater_than_median_arr = is_greater_than_median_arr,
        a = a,
        b = b
    )

    vars += IntermediaryResults(
        is_random_series = vars.a > vars.a_crit and vars.b < vars.b_crit
    )

    return vars