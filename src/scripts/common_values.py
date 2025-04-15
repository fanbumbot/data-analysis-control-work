from pandas import Series

from .intermediary_results import IntermediaryResults

def get_common_values(series: Series):
    vars = IntermediaryResults(
        var = series.var(),
        std = series.std(),
        mean = series.mean()
    )

    return vars