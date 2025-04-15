from pandas import DataFrame, read_excel

from .outliers import get_outliers_three_sigma

TASK1 = "КР Задание 1 в работу.xlsx"
TASK2 = "КР Задание 2 в работу.xlsx"

TASK1_SHEET = "Вариант6"
TASK2_SHEET = "Вариант 6"

DEFAULT_ALPHA = 0.05

task1_dataset: DataFrame = read_excel(TASK1, TASK1_SHEET)

series = task1_dataset['Y']

print('\n')

outliers, all_vars = get_outliers_three_sigma(series, DEFAULT_ALPHA)

print(outliers)