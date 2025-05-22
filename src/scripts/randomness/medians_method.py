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

    total_sign_series = 1
    max_sign_series_size = 0

    current_sign_series_size = 0

    signs_arr = list()

    last_sign = series.loc[0] - median
    for value in series:
        sign = value - median
        signs_arr.append(sign)
        if sign == 0:
            continue

        if ((last_sign == 0) or
            (last_sign > 0 and sign < 0) or
            (last_sign < 0 and sign > 0)
        ):
            total_sign_series += 1
            max_sign_series_size = max(max_sign_series_size, current_sign_series_size)
            current_sign_series_size = 0

        last_sign = sign
        current_sign_series_size += 1

    vars += IntermediaryResults(
        signs_arr = signs_arr,
        a = total_sign_series,
        b = max_sign_series_size
    )

    vars += IntermediaryResults(
        is_random_series = vars.a > vars.a_crit and vars.b < vars.b_crit
    )

    return vars