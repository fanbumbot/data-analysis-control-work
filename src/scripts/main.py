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

task1_dataset.hist()
plt.show()

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


from .append_tests import sign_test, wilcoxon_test, median_random_test, is_normal_dagostino

def test_all(selection):
    factors = ['X1', 'X2', 'X3', 'X4', 'X5', 'Y']
    for factor in factors:
        print(f"{factor} случаен? : {median_random_test(selection[factor])}")

    for factor in factors:
        print(f"У {factor} нормальное распределение? : {is_normal_dagostino(selection[factor])}")

    for i in range(len(factors)):
        for j in range(i+1, len(factors)):
            print(f"{factors[i]} - {factors[j]} однородно (критерий знаков)? : {sign_test(selection[factors[i]], selection[factors[j]])}")
            print(f"{factors[i]} - {factors[j]} однородно (Вилкоксон)? : {wilcoxon_test(selection[factors[i]], selection[factors[j]])}")

test_all(task1_dataset)

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
n_clusters = 3

#--------------------------------------------------------
#scores = list()
#for i in range(2, 10):

kmeans = KMeans(n_clusters=n_clusters, random_state=42)#metric='euclidean', linkage='single')#, random_state=42)
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
    cluster = cluster.drop('labels', axis=1)
    clusters.append(cluster)

x1 = task1_dataset['X2']
y = task1_dataset['Y']
color = task1_dataset['labels'].apply(lambda x: 'r' if x == 0 else'b' if x == 1 else 'y')

plt.scatter(x1, y, c=color)
plt.show()

"""
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
plt.show()"""

import scipy
import scipy.stats

# Кластер для оценки
cluster = clusters[1]
cluster_size = cluster.size

test_all(cluster)

import pingouin as pg

print("Частные коэффициенты корреляции")
print(pg.pcorr(cluster))

cor_matrix = cluster.corr()
d_matrix = cor_matrix.apply(lambda x: x**2)
f_crit = scipy.stats.f.ppf(1-DEFAULT_ALPHA, 1, cluster_size-2)
f_exp_matrix = d_matrix.apply(lambda x: (x/(1-x)) * (cluster_size-2))
significance_matrix = f_exp_matrix.apply(lambda x: x > f_crit)

print("Матрица с корреляцией: ")
print(cor_matrix)
print("Матрица с детерминацией: ")
print(d_matrix)
print("Матрица с F опытным: ")
print(f_exp_matrix)
print("Матрица со значимостью: ")
print(significance_matrix)

import itertools

def get_valid_models(significance_matrix, target='Y'):
    # Шаг 1: Получаем все признаки, значимые для Y
    candidates = significance_matrix.index[significance_matrix[target]].tolist()
    if target in candidates:
        candidates.remove(target)  # исключаем саму Y, если попала

    valid_models = []

    # Шаг 2: Перебираем все возможные непустые подмножества кандидатов
    for r in range(1, len(candidates) + 1):
        for subset in itertools.combinations(candidates, r):
            # Проверка: все признаки в subset незначимо связаны между собой
            is_valid = True
            for i in range(len(subset)):
                for j in range(i + 1, len(subset)):
                    if significance_matrix.loc[subset[i], subset[j]]:
                        is_valid = False
                        break
                if not is_valid:
                    break
            if is_valid:
                valid_models.append(list(subset))

    return valid_models

# Пример использования:
models = get_valid_models(significance_matrix)
for i, model_labels in enumerate(models, 1):
    if len(model_labels) <= 1:
        continue
    model_labels.append('Y')
    print(f"Модель {i}: {model_labels}")

    model = cluster[model_labels]

    # Пусть model — твой DataFrame, y_col — имя зависимой переменной
    y_col = 'Y'
    X_cols = [col for col in model.columns if col != y_col]

    # Извлекаем данные как массивы
    X = model[X_cols].values
    y = model[y_col].values.reshape(-1, 1)  # превращаем в столбец-матрицу

    # Добавляем константу (столбец единиц) для интерсепта
    ones = np.ones((X.shape[0], 1))
    X = np.hstack([ones, X])  # финальная X: [1 x1 x2 ...]

    # Шаги по формуле МНК:
    Xt = X.T                     # X^T
    XtX = Xt @ X                 # X^T * X
    XtX_inv = np.linalg.inv(XtX)  # (X^T * X)^(-1)
    Xty = Xt @ y                 # X^T * y

    beta_hat = XtX_inv @ Xty     # финальные коэффициенты

    # Вывод
    print("Оценённые коэффициенты (включая интерсепт):")
    print(beta_hat)


    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_absolute_error

    # Предсказания: y_pred = X @ beta_hat
    y_pred = X @ beta_hat  # (n_samples, 1)

    # Плоские векторы для метрик (иначе ошибки)
    y_true = y.ravel()
    y_pred_flat = y_pred.ravel()

    # R² и MAE:
    r2 = r2_score(y_true, y_pred_flat)
    mae = mean_absolute_error(y_true, y_pred_flat)

    print(f"R^2: {r2:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")

    n = X.shape[0]         # количество наблюдений
    p = X.shape[1] - 1     # количество признаков (без интерсепта)

    r_adj = d_matrix.apply(lambda x: ((1 - x) * (n - 1)) / (n - p - 1))

    print(f"Скорректированный R^2:")
    print(r_adj)

    n = X.shape[0]         # число наблюдений
    k = X.shape[1]         # число коэффициентов (включая интерсепт)

    # Остатки и RSS
    residuals = y_true - y_pred_flat
    RSS = np.sum(residuals ** 2)

    # AIC и BIC
    AIC = n * np.log(RSS / n) + 2 * k
    BIC = n * np.log(RSS / n) + k * np.log(n)

    print(f"AIC: {AIC:.2f}")
    print(f"BIC: {BIC:.2f}")


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
