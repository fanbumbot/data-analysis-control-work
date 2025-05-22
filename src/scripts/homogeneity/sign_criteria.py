from pandas import Series

from ..intermediary_results import IntermediaryResults

import scipy.stats

def get_sign_criteria_result(series: Series, alpha):
    vars = IntermediaryResults(
        z_crit = scipy.stats.norm.ppf(0.975)
    )

    return vars