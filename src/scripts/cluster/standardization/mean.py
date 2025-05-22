from pandas import Series

from ...intermediary_results import IntermediaryResults

def mean_standardization(data: Series):
    mean = data.mean()

    vars = IntermediaryResults(
        mean = mean
    )

    new_data = [row / mean for row in data]

    vars += IntermediaryResults(
        new_data = new_data
    )

    return vars