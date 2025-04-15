from pandas import Series

from ...intermediary_results import IntermediaryResults

def get_three_sigma_results(series: Series, normalize: bool):
    vars = IntermediaryResults(
        std = series.std(),
        mean = series.mean()
    )

    vars += IntermediaryResults(
        three_sigma = vars.std * 3
    )

    vars += IntermediaryResults(
        left_bound = vars.mean-vars.three_sigma,
        right_bound = vars.mean+vars.three_sigma,
    )

    if normalize:
        if vars.left_bound < 0:
            vars.left_bound = 0
        if vars.right_bound < 0:
            vars.right_bound = 0

    return vars