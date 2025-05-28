from pandas import DataFrame, read_excel
import numpy as np

import math

import matplotlib.pyplot as plt
from pandas import Series

from .common_values import get_common_values

from .intervals import slice_series_by_intervals, optimize_intervals

#TASK1 = "КР Задание 1 в работу.xlsx"
TASK1 = "HousePricePrediction.xlsx"
TASK2 = "КР Задание 2 в работу.xlsx"

TASK1_SHEET = "Sheet1"
TASK2_SHEET = "Вариант 6"

DEFAULT_ALPHA = 0.05

task1_dataset: DataFrame = read_excel(TASK1, TASK1_SHEET)
#task1_dataset = task1_dataset.drop('№', axis=1)
task1_dataset = task1_dataset.drop('Id', axis=1)

#series = task1_dataset['X1']

print('\n')

from .pearson_chi_square_test import get_pearson_test_results

#print(get_pearson_test_results(task1_dataset['X3'], DEFAULT_ALPHA).is_normal_distribution)

task1_dataset['X1'] = task1_dataset['X1'].apply(lambda x: math.log(x+1))
task1_dataset['X5'] = task1_dataset['X5'].apply(lambda x: math.sqrt(x))
task1_dataset['Y'] = task1_dataset['Y'].apply(lambda x: math.log(x+1))

from .outliers import remove_outliers

from scipy.stats import normaltest

task1_dataset['X1'], total_replacements = remove_outliers(task1_dataset['X1'], DEFAULT_ALPHA)
print(total_replacements)
task1_dataset['X2'], total_replacements = remove_outliers(task1_dataset['X2'], DEFAULT_ALPHA)
print(total_replacements)
task1_dataset['X3'], total_replacements = remove_outliers(task1_dataset['X3'], DEFAULT_ALPHA)
print(total_replacements)
task1_dataset['X4'], total_replacements = remove_outliers(task1_dataset['X4'], DEFAULT_ALPHA)
print(total_replacements)
task1_dataset['X5'], total_replacements = remove_outliers(task1_dataset['X5'], DEFAULT_ALPHA)
print(total_replacements)
task1_dataset['Y'], total_replacements = remove_outliers(task1_dataset['Y'], DEFAULT_ALPHA)
print(total_replacements)

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler

scaler = MaxAbsScaler()
task1_dataset_standard = scaler.fit_transform(task1_dataset)

"""
standard_func = lambda x: 1/(x+1.0)

task1_dataset_standard = task1_dataset.copy()
task1_dataset_standard['X1'] = task1_dataset_standard['X1'].apply(standard_func)
task1_dataset_standard['X2'] = task1_dataset_standard['X2'].apply(standard_func)
task1_dataset_standard['X3'] = task1_dataset_standard['X3'].apply(standard_func)
task1_dataset_standard['X4'] = task1_dataset_standard['X4'].apply(standard_func)
task1_dataset_standard['X5'] = task1_dataset_standard['X5'].apply(standard_func)
task1_dataset_standard['Y'] = task1_dataset_standard['Y'].apply(standard_func)
"""
n_clusters = 2

#--------------------------------------------------------
#scores = list()
#for i in range(2, 10):
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(task1_dataset_standard)
labels = kmeans.labels_
    #centroids = kmeans.cluster_centers_
    #print(kmeans.inertia_)

    #score = silhouette_score(task1_dataset_standard, labels)
    #scores.append(score)
#plt.scatter(range(2, 10), scores)

#plt.show()

print(silhouette_score(task1_dataset_standard, labels))

task1_dataset['labels'] = labels

clusters = list()
for i in range(n_clusters):
    cluster = task1_dataset[task1_dataset['labels'] == i]
    clusters.append(cluster)

x1 = task1_dataset['X2']
y = task1_dataset['Y']
color = task1_dataset['labels'].apply(lambda x: 'r' if x == 0 else'b' if x == 1 else 'y')

plt.scatter(x1, y, c=color)
plt.show()

from .randomness.medians_method import get_medians_method_results

for i, cluster in enumerate(clusters):
    print(f"Кластер {i+1} (случайность):")
    print(get_medians_method_results(cluster['X1']).is_random_series)
    print(get_medians_method_results(cluster['X2']).is_random_series)
    print(get_medians_method_results(cluster['X3']).is_random_series)
    print(get_medians_method_results(cluster['X4']).is_random_series)
    print(get_medians_method_results(cluster['X5']).is_random_series)
    print(get_medians_method_results(cluster['Y']).is_random_series)

for i, cluster in enumerate(clusters):
    print(f"Кластер {i+1} (нормальное распределение):")
    s, p = normaltest(cluster['X1'])
    print(p > DEFAULT_ALPHA)
    s, p = normaltest(cluster['X2'])
    print(p > DEFAULT_ALPHA)
    s, p = normaltest(cluster['X3'])
    print(p > DEFAULT_ALPHA)
    s, p = normaltest(cluster['X4'])
    print(p > DEFAULT_ALPHA)
    s, p = normaltest(cluster['X5'])
    print(p > DEFAULT_ALPHA)
    s, p = normaltest(cluster['Y'])
    print(p > DEFAULT_ALPHA)

from scipy.stats import median_test

for factor in ['X1', 'X2', 'X3', 'X4', 'X5', 'Y']:
    stat, p, med, tbl = median_test(*[clusters[i][factor] for i in range(len(clusters))])
    print(f"Неоднородность {factor}: {p < DEFAULT_ALPHA}")

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# 1. Подготовка данных
X = task1_dataset[['X1', 'X2', 'X3', 'X4', 'X5']]
y = task1_dataset['labels']

# Делим на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Обучение классификатора
knn = KNeighborsClassifier(n_neighbors=n_clusters)  # можно поэкспериментировать с k=3, 7 и т.д.
knn.fit(X_train, y_train)

# 3. Предсказание
y_pred = knn.predict(X_test)

# 4. Отчёт
print("Classification report:\n")
print(classification_report(y_test, y_pred))

# 5. Матрица ошибок
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title('Матрица ошибок KNN-классификатора')
plt.xlabel('Предсказанный кластер')
plt.ylabel('Истинный кластер')
plt.tight_layout()
plt.show()

#outliers, all_vars = get_outliers_three_sigma(series, DEFAULT_ALPHA)

#vals = get_common_values(series)

#print(vals)
#print(outliers)

#a = slice_series_by_intervals(series)
#print(a)

#from .pearson_chi_square_test import get_chi_square_critical
#print(get_chi_square_critical(DEFAULT_ALPHA))
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
from .randomness.medians_method import get_medians_method_results

r = get_medians_method_results(task1_dataset['Y'])
for a in r.signs_arr:
    print(a)

print('\n')
print(r.a, r.a_crit)

task1_dataset['Y'].hist()
plt.show()"""

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
