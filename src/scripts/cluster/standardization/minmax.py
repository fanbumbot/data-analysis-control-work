from pandas import Series

from ...intermediary_results import IntermediaryResults

def minmax_standardization(data: Series):
    min = data.min()
    max = data.max()
    mean = data.mean()

    vars = IntermediaryResults(
        min = min,
        max = max,
        mean = mean
    )

    new_data = [abs(row-mean)/(max-min) for row in data]

    vars += IntermediaryResults(
        new_data = new_data
    )

    return vars