from dataclasses import dataclass
from typing import Any, Iterator
from collections import OrderedDict
import math

from pandas import Series, concat

from .intermediary_results import IntermediaryResults

@dataclass(frozen=True)
class Interval:
    start: Any
    end: Any

    def __post_init__(self):
        if self.start > self.end:
            raise ValueError
        
class SeriesByIntervals(OrderedDict):
    def __getitem__(self, key: Interval):
        return super().__getitem__(key)
    
    def __setitem__(self, key: Interval, value: Series):
        return super().__setitem__(key, value)
    
    def __delitem__(self, key: Interval):
        return super().__delitem__(key)

    def __iter__(self) -> Iterator[tuple[Interval, Series]]:
        return super().__iter__()

def get_interval_count_results(size: int, round_result: bool = True):
    vars = IntermediaryResults(
        interval_count = 1 + 3.32 * math.log10(size)
    )

    if round_result:
        vars.interval_count = math.ceil(vars.interval_count)

    return vars

def get_interval_distance_by_count(x_max: float, x_min: float, interval_count: int):
    vars = IntermediaryResults(
        full_distance = x_max - x_min
    )
    
    vars += IntermediaryResults(
        interval_distance = vars.full_distance/interval_count
    )

    return vars

def slice_series_by_intervals_raw(series: Series):
    vars = IntermediaryResults(
        max = series.max(),
        min = series.min(),
        size = series.size
    )
    
    vars += get_interval_count_results(vars.size)
    vars += get_interval_distance_by_count(vars.max, vars.min, vars.interval_count)

    series_by_intervals = SeriesByIntervals()

    current = vars.min
    next = current + vars.interval_distance
    for _ in range(vars.interval_count):
        interval = Interval(current, next)
        all_values_in_interval = series[(series >= current) & (series < next)]

        series_by_intervals[interval] = all_values_in_interval
        series_by_intervals.move_to_end(interval)

        current = next
        next += vars.interval_distance

    # Добавляем максимальное значение
    all_values_in_interval = concat((series_by_intervals[interval], series[series >= next]))
    series_by_intervals[interval] = all_values_in_interval
    series_by_intervals.move_to_end(interval)

    vars += IntermediaryResults(
        series_by_intervals = series_by_intervals
    )

    return vars

def optimize_intervals(series_by_intervals: SeriesByIntervals):
    new_series_by_intervals = SeriesByIntervals()

    for interval, series in series_by_intervals.items():
        new_series_by_intervals[interval] = series
        new_series_by_intervals.move_to_end(interval)

    #for interval, series in new_series_by_intervals:
    #    if series.size == 0:

def slice_series_by_intervals(series: Series):
    results = slice_series_by_intervals_raw(series)

    return results


