import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

from ..intermediary_results import IntermediaryResults

def apply_kmeans(data, clusters_count: int):
    kmeans = KMeans(n_clusters=clusters_count, random_state=42)
    kmeans.fit(data)
    return kmeans

def get_wcss(kmeans: KMeans):
    return kmeans.inertia_

def get_silhouette_score(data, kmeans: KMeans):
    return silhouette_score(data, kmeans.fit_predict(data))

def compute_kmeans_and_score(data, clusters_count: int):
    kmeans = apply_kmeans(data, clusters_count)
    wcss = get_wcss(kmeans)
    silhouette_score = get_silhouette_score(data, kmeans)
    return kmeans, wcss, silhouette_score

def compute_most_successful_clusters_by_count(data, clusters_count: int, iterations: int):
    most_successful_kmeans: KMeans = None
    most_successful_silhouette_score = 0
    most_successful_wcss = 0
    for i in range(iterations):
        kmeans, wcss, silhouette_score = compute_kmeans_and_score(data, clusters_count)
        if wcss > most_successful_wcss and silhouette_score > most_successful_silhouette_score:
            most_successful_kmeans = kmeans
            most_successful_wcss = wcss
            most_successful_silhouette_score = silhouette_score
    return most_successful_kmeans, most_successful_wcss, most_successful_silhouette_score

def compute_most_successful_clusters(data, min_clusters: int, max_clusters: int, max_iterations: int):
    answer = list()
    for clusters_count in range(min_clusters, max_clusters+1):
        kmeans, wcss, silhouette_score = compute_most_successful_clusters_by_count(
            data, clusters_count, max_iterations
        )
        answer.append((kmeans, wcss, silhouette_score))
    return answer