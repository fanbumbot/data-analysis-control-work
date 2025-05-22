from pandas import Series

from ...intermediary_results import IntermediaryResults

def max_standardization(data: Series):
    max = data.max()

    vars = IntermediaryResults(
        max = max
    )

    new_data = [row / max for row in data]

    vars += IntermediaryResults(
        new_data = new_data
    )

    return vars