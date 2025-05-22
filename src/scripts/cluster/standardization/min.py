from pandas import Series

from ...intermediary_results import IntermediaryResults

def min_standardization(data: Series):
    min = data.min()

    vars = IntermediaryResults(
        min = min
    )

    new_data = [row / min for row in data]

    vars += IntermediaryResults(
        new_data = new_data
    )

    return vars