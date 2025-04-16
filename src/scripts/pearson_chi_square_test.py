from typing import Iterator

from pandas import Series

from scipy import stats as stats

from .intermediary_results import IntermediaryResults

from .intervals import slice_series_by_intervals, SeriesByIntervals, Interval

def get_chi_square_critical(k, alpha):
    return stats.chi2.ppf(1-alpha, k)

class IntervalPearsonTestResults(SeriesByIntervals):
    def __getitem__(self, key: Interval):
        return super().__getitem__(key)
    
    def __setitem__(self, key: Interval, value: IntermediaryResults):
        return super().__setitem__(key, value)
    
    def __delitem__(self, key: Interval):
        return super().__delitem__(key)

    def __iter__(self) -> Iterator[tuple[Interval, IntermediaryResults]]:
        return super().__iter__()

def get_pearson_test_results(series: Series, alpha):
    vars = slice_series_by_intervals(series)

    vars += IntermediaryResults(
        mean = series.mean(),
        std = series.std(),
        size = series.size,
        chi_square_critical = get_chi_square_critical(vars.interval_count-1-1, alpha)
    )

    series_by_intervals: SeriesByIntervals = vars.series_by_intervals
    chi_exp = 0

    results_by_intervals = IntervalPearsonTestResults()

    for interval, sub_series in series_by_intervals.items():
        interval_start = interval.start
        interval_end = interval.end
        F_start = stats.norm.cdf((interval_start-vars.mean)/vars.std)
        F_end = stats.norm.cdf((interval_end-vars.mean)/vars.std)
        p = F_end-F_start
        size_estimation = vars.size * p
        chi_exp_k = (sub_series.size - size_estimation)**2/size_estimation
        chi_exp += chi_exp_k

        vars_k = IntermediaryResults(
            interval_start = interval_start,
            interval_end = interval_end,
            F_start = F_start,
            F_end = F_end,
            p = p,
            size_estimation = size_estimation,
            chi_exp_k = chi_exp_k,
            current_chi_exp = chi_exp
        )
        results_by_intervals[interval] = vars_k
        results_by_intervals.move_to_end(interval)

    vars += IntermediaryResults(
        results_by_intervals = results_by_intervals,
        chi_exp = chi_exp
    )

    vars += IntermediaryResults(
        is_normal_distribution = vars.chi_exp < vars.chi_square_critical
    )

    return vars