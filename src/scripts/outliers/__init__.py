from pandas import Series

from .grubbs_method import get_outliers_grubbs
from .iqr_method import get_outliers_iqr
from .three_sigma_method import get_outliers_three_sigma