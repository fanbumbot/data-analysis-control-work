from pandas import DataFrame, read_excel

import matplotlib.pyplot as plt

from .outliers import get_outliers_three_sigma

from .common_values import get_common_values

from .intervals import slice_series_by_intervals, optimize_intervals

TASK1 = "КР Задание 1 в работу.xlsx"
TASK2 = "КР Задание 2 в работу.xlsx"

TASK1_SHEET = "Вариант6"
TASK2_SHEET = "Вариант 6"

DEFAULT_ALPHA = 0.05

task1_dataset: DataFrame = read_excel(TASK1, TASK1_SHEET)

series = task1_dataset['X1']

print('\n')

#outliers, all_vars = get_outliers_three_sigma(series, DEFAULT_ALPHA)

#vals = get_common_values(series)

#print(vals)
#print(outliers)

#a = slice_series_by_intervals(series)
#print(a)

#from .pearson_chi_square_test import get_chi_square_critical
#print(get_chi_square_critical(DEFAULT_ALPHA))

from .pearson_chi_square_test import get_pearson_test_results

#vars = get_pearson_test_results(series, DEFAULT_ALPHA)

#print(vars)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import scipy
import scipy.cluster
from sklearn.metrics import silhouette_score
data = scipy.cluster.vq.whiten(task1_dataset)

iterator = range(2, 20)

scores = list()
wcss = list()
for i in iterator:
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(task1_dataset)
    labels = kmeans.labels_
    wcss.append(kmeans.inertia_)

    score = silhouette_score(task1_dataset, kmeans.fit_predict(task1_dataset))
    scores.append(score)

plt.plot(list(iterator), scores)
#plt.plot(list(iterator), wcss)
plt.show()

"""
plt.scatter(data[:, 0], data[:, 4], c=labels, s=50, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 4], c='red', s=200, marker='X', label='Центры кластеров')
plt.title("Результаты кластеризации методом k-средних")
plt.xlabel("Признак 1")
plt.ylabel("Признак 2")
plt.legend()
plt.show()
"""