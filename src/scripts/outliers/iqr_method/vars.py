from pandas import Series

from ...intermediary_vars import IntermediaryVariables

def get_iqr_vars(series: Series, normalize: bool):
    vars = IntermediaryVariables(
        q1 = series.quantile(0.25),
        q3 = series.quantile(0.75)
    )

    vars += IntermediaryVariables(
        iqr = vars.q3 - vars.q1
    )

    vars += IntermediaryVariables(
        left_bound = vars.q1 - 1.5 * vars.iqr,
        right_bound = vars.q3 + 1.5 * vars.iqr
    )

    if normalize:
        if vars.left_bound < 0:
            vars.left_bound = 0
        if vars.right_bound < 0:
            vars.right_bound = 0

    return vars