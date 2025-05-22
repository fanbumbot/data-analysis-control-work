from pandas import DataFrame, read_excel

import matplotlib.pyplot as plt

from .common_values import get_common_values

from .intervals import slice_series_by_intervals, optimize_intervals

TASK1 = "КР Задание 1 в работу.xlsx"
TASK2 = "КР Задание 2 в работу.xlsx"

TASK1_SHEET = "Вариант6"
TASK2_SHEET = "Вариант 6"

DEFAULT_ALPHA = 0.05

task1_dataset: DataFrame = read_excel(TASK1, TASK1_SHEET)
task1_dataset = task1_dataset.drop('№', axis=1)

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
"""
from .cluster.kmeans import compute_most_successful_clusters
import scipy.cluster

data = scipy.cluster.vq.whiten(task1_dataset)

ans = compute_most_successful_clusters(data, 2, 20, 1)

scores = [arr[1] for arr in ans]
print(ans)

plt.plot(list(range(2, 21)), scores)
#plt.plot(list(iterator), wcss)
plt.show()"""

"""import numpy as np
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
plt.show()"""

"""
plt.scatter(data[:, 0], data[:, 4], c=labels, s=50, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 4], c='red', s=200, marker='X', label='Центры кластеров')
plt.title("Результаты кластеризации методом k-средних")
plt.xlabel("Признак 1")
plt.ylabel("Признак 2")
plt.legend()
plt.show()
"""

"""
from .outliers import get_outliers_grubbs, get_outliers_iqr, get_outliers_three_sigma

def a(series):
    outliers1, _, _ = get_outliers_grubbs(series, DEFAULT_ALPHA)
    outliers2, _ = get_outliers_iqr(series, True)
    outliers3, _ = get_outliers_three_sigma(series, True)

    print(outliers1)
    print(outliers2)
    print(outliers3)

print("X1:")
a(task1_dataset['X1'])
print("X2:")
a(task1_dataset['X2'])
print("X3:")
a(task1_dataset['X3'])
print("X4:")
a(task1_dataset['X4'])
print("Y:")
a(task1_dataset['Y'])"""
"""
from .randomness.medians_method import get_medians_method_results
from pandas import Series

vars = get_medians_method_results(task1_dataset['Y'])

print(vars)
print(vars.is_random_series)"""

"""
from .homogeneity.sign_criteria import get_sign_criteria_result

vars = get_sign_criteria_result(task1_dataset['Y'], DEFAULT_ALPHA)

#print(vars)

import scipy.stats
a = scipy.stats.mannwhitneyu(task1_dataset['X3'], task1_dataset['X4'])

plt.hist(task1_dataset['X4'])
plt.show()"""

"""
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from .cluster.kmeans import compute_kmeans_and_score

scaler = MinMaxScaler()
data = task1_dataset
data_scaled = scaler.fit_transform(data)

all_wcss = list()

for i in range(2, 41):
    #kmeans, wcss, silhouette_score = compute_kmeans_and_score(data_scaled, i)
    clustering = AgglomerativeClustering(i)
    clustering.fit(data_scaled)

    score = silhouette_score(data_scaled, clustering.labels_)
    #wcss = clustering.
    all_wcss.append(score)

plt.plot(list(range(2, 41)), all_wcss)

plt.show()
"""
"""
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering

from .cluster.kmeans import compute_kmeans_and_score

scaler = MinMaxScaler()
data = task1_dataset

data_scaled = scaler.fit_transform(data)

#kmeans, wcss, silhouette_score = compute_kmeans_and_score(data_scaled, 6)
clustering = AgglomerativeClustering(6)
clustering.fit(data_scaled)

print(len(data_scaled))

for i in range(len(data)):
    cluster = clustering.labels_[i]

    color = 0

    match cluster:
        case 0:
            color = 'r'
        case 1:
            color = 'g'
        case 2:
            color = 'b'
        case 3:
            color = 'yellow'
        case 4:
            color = 'purple'
        case 5:
            color = 'black'

    plt.scatter(data_scaled[i][1], data_scaled[i][4], c=color)

plt.show()"""

#from .outliers import get_outliers_three_sigma

#outliers, vars = get_outliers_three_sigma(task1_dataset['Y'], True)
#print(outliers)
#print(vars)
"""
import scipy.stats

a = scipy.stats.shapiro(task1_dataset['Y'])

print(a.pvalue > DEFAULT_ALPHA)
"""

#from .outliers import get_outliers_grubbs

#outliers, vars, _ = get_outliers_grubbs(task1_dataset['Y'], DEFAULT_ALPHA)
#print(len(outliers))


"""
import pingouin

print(task1_dataset.pcorr())


from .outliers import remove_outliers

from .cluster.hierarchy import make_hierarchy_clustering, make_hierarchy_clustering_ward
from .cluster.standardization.minmax import minmax_standardization
from .cluster.standardization.max import max_standardization
from .cluster.standardization.mean import mean_standardization
from .cluster.standardization.std import std_standardization

import numpy as np

x1_without_outliers, _ = remove_outliers(task1_dataset['X1'], DEFAULT_ALPHA)
x2_without_outliers, _ = remove_outliers(task1_dataset['X2'], DEFAULT_ALPHA)
x3_without_outliers, _ = remove_outliers(task1_dataset['X3'], DEFAULT_ALPHA)
x4_without_outliers, _ = remove_outliers(task1_dataset['X4'], DEFAULT_ALPHA)
y_without_outliers, l = remove_outliers(task1_dataset['Y'], DEFAULT_ALPHA)
print(l)
print(y_without_outliers)

standardization_func = max_standardization



x1_st = standardization_func(x1_without_outliers).new_data
x2_st = standardization_func(x2_without_outliers).new_data
x3_st = standardization_func(x3_without_outliers).new_data
x4_st = standardization_func(x4_without_outliers).new_data
y_st = standardization_func(y_without_outliers).new_data

all_param = [x1_st, x2_st, x3_st, x4_st, y_st]

clusters1, silhouettes1 = make_hierarchy_clustering_ward(
    all_param)
plt.plot(
    list(range(2, 26)), silhouettes1[1:25],
    label="Метод Варда"
)

clusters2, silhouettes2 = make_hierarchy_clustering(
    all_param,
    get_dist_func=lambda x, y: np.linalg.norm(x-y))
plt.plot(
    list(range(2, 26)), silhouettes2[1:25],
    label="Евклидово расстояние"
)

clusters3, silhouettes3 = make_hierarchy_clustering(
    all_param,
    get_dist_func=lambda x, y: sum(abs(x-y)))
plt.plot(
    list(range(2, 26)), silhouettes3[1:25],
    label="Манхеттенское расстояние"
)

clusters4, silhouettes4 = make_hierarchy_clustering(
    all_param, 1,
    get_dist_func=lambda x, y: 1 - (np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y))))
plt.plot(
    list(range(2, 26)), silhouettes4[1:25],
    label="Косинусное расстояние"
)

plt.legend()
plt.show()
"""

from .randomness.medians_method import get_medians_method_results

r = get_medians_method_results(task1_dataset['Y'])
for a in r.signs_arr:
    print(a)

print('\n')
print(r.a, r.a_crit)

task1_dataset['Y'].hist()
plt.show()

#print([len(clusters.points) for clusters in clusters4.clusters])

#print([len(cluster) for cluster in clusters1])
#print()
#print([len(cluster) for cluster in clusters2])


#plt.plot(list(range(2, 26)), silhouettes1[1:25])
#plt.show()


"""
print([len(cluster) for cluster in clusters1])
print()
print([len(cluster) for cluster in clusters2])


for cluster_index in range(len(clusters1)):
    cluster = clusters1[cluster_index]

    if cluster_index == 0:
        color = "r"
    elif cluster_index == 1:
        color = "g"
    elif cluster_index == 2:
        color = "b"
    elif cluster_index == 3:
        color = "yellow"
    elif cluster_index == 4:
        color = "skyblue"

    for point in cluster:
        plt.scatter(point[1], point[4], c=color)

plt.show()
"""
