from pandas import Series

from ...intermediary_results import IntermediaryResults

def get_iqr_results(series: Series, normalize: bool):
    vars = IntermediaryResults(
        q1 = series.quantile(0.25),
        q3 = series.quantile(0.75)
    )

    vars += IntermediaryResults(
        iqr = vars.q3 - vars.q1
    )

    vars += IntermediaryResults(
        left_bound = vars.q1 - 1.5 * vars.iqr,
        right_bound = vars.q3 + 1.5 * vars.iqr
    )

    if normalize:
        if vars.left_bound < 0:
            vars.left_bound = 0
        if vars.right_bound < 0:
            vars.right_bound = 0

    return vars