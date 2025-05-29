import numpy as np
import scipy.stats
from scipy.stats import normaltest, binomtest, wilcoxon

DEFAULT_ALPHA = 0.05

# --- Функция проверки случайности (метод медиан) ---
def median_random_test(series, alpha=DEFAULT_ALPHA):
    median = np.median(series)
    sequence = [1 if x > median else 0 for x in series]
    
    runs = 1
    for i in range(1, len(sequence)):
        if sequence[i] != sequence[i - 1]:
            runs += 1

    n1 = sum(sequence)
    n2 = len(sequence) - n1

    if n1 == 0 or n2 == 0:
        return False  # все значения выше или ниже медианы — явно не случайно

    expected_runs = 1 + (2 * n1 * n2) / (n1 + n2)
    std_runs = np.sqrt((2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) /
                       (((n1 + n2) ** 2) * (n1 + n2 - 1)))
    
    z = (runs - expected_runs) / std_runs
    from scipy.stats import norm
    p_value = 2 * (1 - norm.cdf(abs(z)))
    
    return p_value > alpha

# --- Тест на нормальность ---
def is_normal_dagostino(x, alpha=DEFAULT_ALPHA):
    stat, p = normaltest(x)
    return p > alpha

# --- Критерий знаков ---
def sign_test(x, y, alpha=DEFAULT_ALPHA):
    diff = np.array(x) - np.array(y)
    signs = diff[diff != 0]
    if len(signs) == 0:
        return True
    n_positive = np.sum(signs > 0)
    n = len(signs)
    p = binomtest(n_positive, n=n, p=0.5, alternative='two-sided')
    return p.pvalue > alpha

# --- Критерий Вилкоксона ---
def wilcoxon_test(x, y, alpha=DEFAULT_ALPHA):
    try:
        stat, p = wilcoxon(x, y)
        return p > alpha
    except ValueError:
        return False