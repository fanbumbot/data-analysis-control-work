from pandas import Series

from ...intermediary_results import IntermediaryResults

def std_standardization(data: Series):
    std = data.std()
    mean = data.mean()

    vars = IntermediaryResults(
        std = std,
        mean = mean
    )

    new_data = [abs(row-mean)/std for row in data]

    vars += IntermediaryResults(
        new_data = new_data
    )

    return vars