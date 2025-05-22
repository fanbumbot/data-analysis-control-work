from pandas import Series

import numpy as np

class ClusterPoint:
    def __init__(
        self,
        dims: np.ndarray,
        index
    ):
        self.dims = dims
        self.silhouette_score = None
        self.index = index

    def __len__(self):
        return len(self.dims)

    @classmethod
    def get_mean_dist(main_point_dims: np.ndarray, other_points_dims: list[np.ndarray], get_dist_func):
        distances = np.array([
                get_dist_func(main_point_dims, point_dims)
            for point_dims in other_points_dims
        ])

        return distances.mean()

    def calculate_silhouette_score(self, my_cluster: "Cluster", other_clusters: list["Cluster"], get_dist_func):
        if my_cluster == None or other_clusters == None or get_dist_func == None:
            return
        
        a = ClusterPoint.get_mean_dist(self.dims, (point.dims for point in my_cluster.points), get_dist_func)

        b = np.inf
        for cluster in other_clusters:
            b_i = ClusterPoint.get_mean_dist(self.dims, (point.dims for point in cluster.points), get_dist_func)
            b = min(b, b_i)

        s = (b-a)/max(a, b)

        return s

class Cluster:
    def __init__(
        self,
        points: list[ClusterPoint]
    ):
        self.points = points
        self.dimensions = len(points[0])
        self.centroid = self.calculate_centroid()
        self.silhouette_score = None

    def __len__(self):
        return len(self.points)

    def calculate_centroid(self):
        points_count = len(self)
        if points_count == 0:
            return None

        dimensions = self.dimensions
        points = self.points

        centroid = list()
        
        for d in range(dimensions):
            summ = 0

            for p in range(points_count):
                summ += points[p].dims[d]

            mean = summ/points_count
            centroid.append(mean)

        return np.array(centroid).astype(float)
    
class ClustersSet:
    def __init__(
        self,
        clusters: list[Cluster],
        get_dist_func
    ):
        self.clusters = clusters
        self.get_dist_func = get_dist_func

        matrix_order = len(clusters)

        self.cluster_distance_matrix = DistanceMatrix(
            matrix_order,
            ClustersSet.get_distances_between_points(
                ClustersSet.get_all_points_from_clusters(clusters),
                get_dist_func
            )
        )

        self.distance_matrix = DistanceMatrix(
            matrix_order,
            ClustersSet.get_distances_between_points(
                ClustersSet.get_all_points_from_clusters(clusters),
                get_dist_func
            )
        )

    def __len__(self):
        return len(self.clusters)
    
    @classmethod
    def get_all_points_from_clusters(cls, clusters: list[Cluster]):
        return [
            point.dims for cluster in clusters for point in cluster.points
        ]
    
    @classmethod
    def get_distances_between_point_and_points(
        cls,
        point_dims: np.ndarray,
        points_dims: list[np.ndarray],
        get_dist_func
    ):
        distances = np.array([get_dist_func(point_dims, sub_point_dim) for sub_point_dim in points_dims])
        return distances
    
    @classmethod
    def get_distances_between_points(
        cls,
        points_dims: list[np.ndarray],
        get_dist_func
    ):
        length = len(points_dims)
        distances = np.array([[
                    get_dist_func(points_dims[i], points_dims[j])
                for i in range(length)]
            for j in range(length)
        ])

        return distances
    
    def get_distances_between_point_and_centroids(
        self,
        point_dims: np.ndarray,
    ):
        distances = ClustersSet.get_distances_between_point_and_points(
            point_dims,
            (cluster.centroid for cluster in self.clusters),
            self.get_dist_func
        )
        return distances
    
    def get_distances(self, cluster: Cluster):
        distances = self.get_distances_between_point_and_centroids(
            cluster.centroid
        )
        return distances
    
    def combine_clusters_by_min(self):
        _, min_i, min_j = self.cluster_distance_matrix.get_min()

        if min_i > min_j:
            min_i, min_j = min_j, min_i

        cluster1 = self.clusters[min_i]
        cluster2 = self.clusters[min_j]

        points = cluster1.points

        points.extend(cluster2.points)
        new_cluster = Cluster(points)

        del self.clusters[min_j]
        del self.clusters[min_i]
        self.clusters.append(new_cluster)

        distances = self.get_distances(new_cluster)

        self.cluster_distance_matrix.delete_by_index(min_j)
        self.cluster_distance_matrix.delete_by_index(min_i)
        self.cluster_distance_matrix.add_in_end(distances)

    def get_silhouette_score(self):
        distance_matrix = self.distance_matrix.matrix

        s_sum = 0
        for main_cluster_i, main_cluster in enumerate(self.clusters):
            for main_point_i, main_point in enumerate(main_cluster.points):
                if len(main_cluster.points) <= 1:
                    a_i = 0
                else:
                    a_sum = 0
                    for other_point_i, other_point in enumerate(main_cluster.points):
                        if main_point_i != other_point_i:
                            a_sum += distance_matrix[main_point.index][other_point.index]
                    a_i = a_sum/(len(main_cluster.points)-1)

                b_i = np.inf
                for other_cluster_i, other_cluster in enumerate(self.clusters):
                    if other_cluster_i != main_cluster_i:
                        b_sum = 0
                        for other_point_i, other_point in enumerate(other_cluster.points):
                            b_sum += distance_matrix[main_point.index][other_point.index]
                        b_mean = b_sum/len(other_cluster)
                        if b_mean < b_i:
                            b_i = b_mean
                
                s_i = (b_i-a_i)/max(a_i, b_i)
                s_sum += s_i
        s = s_sum/len(distance_matrix)

        return s

class ClustersSetWard(ClustersSet):
    def __init__(self, clusters):
        super().__init__(clusters, lambda x, y: np.linalg.norm(x-y))

        self.cluster_distance_matrix = DistanceMatrix(
            len(clusters),
            [[
                        ClustersSetWard.get_ward_distance(cluster1, cluster2)
                    for cluster1 in clusters]
                for cluster2 in clusters
            ]
        )

    @classmethod
    def get_ward_distance(cls, cluster1: Cluster, cluster2: Cluster):
        centroid1 = cluster1.centroid
        centroid2 = cluster2.centroid
        n1 = len(cluster1)
        n2 = len(cluster2)
        
        distance = (n1 * n2) / (n1 + n2) * np.sum((centroid1 - centroid2)**2)
        return distance

    def get_distances(self, cluster1: Cluster):
        return np.array([
                ClustersSetWard.get_ward_distance(cluster1, cluster2)
            for cluster2 in self.clusters
        ])

class DistanceMatrix:
    def __init__(
        self,
        start_order,
        matrix = None
    ):
        self.minimum = None
        self.minimum_i: int = None
        self.minimum_j: int = None

        if matrix is None:
            self.matrix = np.array([[0 for _ in range(start_order)] for _ in range(start_order)])
        else:
            self.matrix = matrix

        self.minimum, self.minimum_i, self.minimum_j = self.__get_min()

    def delete_by_index(self, index: int):
        self.matrix = np.delete(np.delete(self.matrix, index, axis=0), index, axis=1)

        if index == self.minimum_i:
            self.minimum, self.minimum_i, self.minimum_j = self.__get_min()
        elif index == self.minimum_j:
            self.minimum, self.minimum_i, self.minimum_j = self.__get_min()
        elif index < self.minimum_i:
            self.minimum_i -= 1
        elif index < self.minimum_j:
            self.minimum_j -= 1

    def add_in_end(self, distances: np.ndarray):
        #self.matrix = np.hstack([np.vstack([self.matrix, distances[:-1]]), distances.reshape(-1, 1)])

        order = self.get_order()
        distances_without_null = distances[:order]
        self.matrix = np.vstack([self.matrix, distances_without_null])
        self.matrix = np.hstack([self.matrix, distances.reshape(-1, 1)])

        if len(distances_without_null) == 0:
            return

        min_i = int(distances_without_null.argmin())
        minimum = distances_without_null[min_i]
        if minimum < self.minimum:
            self.minimum = minimum
            self.minimum_i = min_i
            self.minimum_j = order
    
    def get_min(self):
        return self.minimum, self.minimum_i, self.minimum_j

    def get_order(self):
        return len(self.matrix)

    def __get_min(self):
        order = self.get_order()

        if order == 1:
            return self.matrix[0][0], 0, 0

        minimum_i: int = None
        minimum_j: int = None
        minimum: float = np.inf
        
        for i in range(order):
            for j in range(i+1, order):
                value = self.matrix[i][j]
                if value < minimum:
                    minimum = value
                    minimum_i = i
                    minimum_j = j

        return minimum, minimum_i, minimum_j

def make_hierarchy_clustering(dataset: list[Series], min_clusters: int = 1, get_dist_func = None):
    dimensions = len(dataset)
    length = len(dataset[0])

    points = np.array([[
                dataset[point_dimension][point_i]
            for point_dimension in range(dimensions)]
        for point_i in range(length)
    ])

    clusters_list: list[Cluster] = list()
    for i, point_dims in enumerate(points):
        point = ClusterPoint(point_dims, i)
        cluster = Cluster([point])
        clusters_list.append(cluster)

    clusters = ClustersSet(clusters_list, get_dist_func)

    silhouettes = list()
    silhouette = clusters.get_silhouette_score()
    silhouettes.append(silhouette)

    while len(clusters) > min_clusters:
        clusters.combine_clusters_by_min()

        silhouette = clusters.get_silhouette_score()
        silhouettes.append(silhouette)

    silhouettes.reverse()

    return clusters, silhouettes

def make_hierarchy_clustering_ward(dataset: list[Series], min_clusters: int = 1):
    dimensions = len(dataset)
    length = len(dataset[0])

    points = np.array([[
                dataset[point_dimension][point_i]
            for point_dimension in range(dimensions)]
        for point_i in range(length)
    ])

    clusters_list: list[Cluster] = list()
    for i, point_dims in enumerate(points):
        point = ClusterPoint(point_dims, i)
        cluster = Cluster([point])
        clusters_list.append(cluster)

    clusters = ClustersSetWard(clusters_list)

    silhouettes = list()

    while len(clusters) > min_clusters:
        clusters.combine_clusters_by_min()
        silhouette = clusters.get_silhouette_score()

        silhouettes.append(silhouette)

    silhouettes.reverse()

    return clusters, silhouettes