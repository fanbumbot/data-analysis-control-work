from pandas import Series

from ...intermediary_vars import IntermediaryVariables

def get_three_sigma_vars(series: Series, normalize: bool):
    vars = IntermediaryVariables(
        std = series.std(),
        mean = series.mean()
    )

    vars += IntermediaryVariables(
        three_sigma = vars.std * 3
    )

    vars += IntermediaryVariables(
        left_bound = vars.mean-vars.three_sigma,
        right_bound = vars.mean+vars.three_sigma,
    )

    if normalize:
        if vars.left_bound < 0:
            vars.left_bound = 0
        if vars.right_bound < 0:
            vars.right_bound = 0

    return vars