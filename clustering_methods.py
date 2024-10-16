from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import numpy as np
import random
import torch

feature_dim = 2048

from logger import logging, CustomLogger

clmLogger = CustomLogger('clm')

def print_memory_usage(message=""):
    """Print GPU memory usage."""
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    clmLogger.log(f"{message} - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")


def getClusterCenters(data, labels, batch_size, custom_distance):
    unique_labels = np.unique(labels)
    clmLogger.log("unique_labels {}".format(unique_labels))
    medoid_indexes = []
    with torch.no_grad():
        for label in unique_labels:
            clmLogger.log("------LABEL {}".format(label))
            cluster_indexes = np.where(labels == label)[0]
            cluster_blob = data[cluster_indexes]
            num_points = cluster_blob.shape[0]
            print('Cluster blob initial shape ', cluster_blob.shape)
            cluster_blob = torch.from_numpy(data[cluster_indexes].reshape(num_points,-1,feature_dim)).float() 
            print('Cluster blob shape 2 ', cluster_blob.shape)

            # Calculate distance matrix (upper triangle since direction doesn't matter)
            distance_matrix = np.zeros((num_points, num_points))
            for i in range(num_points):
                for j in range(i+1, num_points, batch_size):
                    end = min(j+batch_size, num_points)
                    batch_videos_j = cluster_blob[j:end].to('cuda')
                    batch_videos_i = cluster_blob[i].unsqueeze(0).expand(end-j, -1, -1).to('cuda')
                    distance = custom_distance(batch_videos_i, batch_videos_j).cpu().numpy()
                    distance_matrix[i, j:end] = distance
                    distance_matrix[j:end, i] = distance  # Symmetric matrix
            total_distances = np.sum(distance_matrix, axis=1)
            medoid_index = np.argmin(total_distances)
            clmLogger.log("medoid_index {}".format(medoid_index))
            final_index = cluster_indexes[medoid_index]
            medoid_indexes.append(final_index)
    medoid_indexes = np.array(medoid_indexes)
    clmLogger.log('Medoid indexes {}'.format(medoid_indexes))
    return medoid_indexes, data[medoid_indexes]

def getClusterCentersWDistance(data, labels, precomp_distance):
    unique_labels = np.unique(labels)
    clmLogger.log("unique_labels {}".format(unique_labels))
    medoid_indexes = []
    with torch.no_grad():
        for label in unique_labels:
            clmLogger.log("------LABEL {}".format(label))
            cluster_indexes = np.where(labels == label)[0]
            cluster_blob = data[cluster_indexes]
            num_points = cluster_blob.shape[0]
            distance_matrix = np.zeros((num_points, num_points))
            for i in range(num_points):
                for j in range(i+1, num_points):
                    distance_matrix[i,j] = precomp_distance[cluster_indexes[i], cluster_indexes[j]]
            total_distances = np.sum(distance_matrix, axis=1)
            medoid_index = np.argmin(total_distances)
            clmLogger.log("medoid_index {}".format(medoid_index))
            final_index = cluster_indexes[medoid_index]
            medoid_indexes.append(final_index)
    medoid_indexes = np.array(medoid_indexes)
    clmLogger.log('Medoid indexes {}'.format(medoid_indexes))
    return medoid_indexes, data[medoid_indexes]




def visualize_distance_matrix(distance_matrix):
    plt.imshow(distance_matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Distance')
    plt.title('Distance Matrix Heatmap')
    plt.xlabel('Data Point Index')
    plt.ylabel('Data Point Index')
    plt.show()

def visualize_clusters_with_pca(data, labels, title="DBSCAN Clustering", method='pca'):
    """
    Function to visualize clustered data with PCA or t-SNE, with each label having a different color.
    
    Parameters:
    - data: numpy array, the dataset (samples, 300, 1024)
    - labels: numpy array, the cluster labels from DBSCAN
    - method: 'pca' or 'tsne' for dimensionality reduction
    """
    # Flatten the data from (samples, 300, 1024) to (samples, 300 * 1024)
    clmLogger.log('Data size {}'.format(data.shape))
    flattened_data = data
    #flattened_data = data.reshape(data.shape[0], -1)
    
    # Apply dimensionality reduction (PCA or t-SNE)
    if method == 'pca':
        reducer = PCA(n_components=2)  # Reduce to 2 dimensions
        reduced_data = reducer.fit_transform(flattened_data)
    elif method == 'tsne':
        # First PCA in temporal dimension, then TSNE IN REST
        data_shortened = np.zeros((data.shape[0],10,1024))
        reducer = PCA(n_components=10)
        for i in range(len(data)):
            vid = data[i].reshape(1024,-1)
            reduced_data = reducer.fit_transform(vid)
            data_shortened[i] = reduced_data.reshape(-1,1024)
        data_shortened = data_shortened.reshape(-1, 10*1024)
        perplexity_value = min(30, len(data_shortened) - 1)  # Ensure perplexity < n_samples
        reducer = TSNE(n_components=2, perplexity=perplexity_value, random_state=0)
        reduced_data = reducer.fit_transform(data_shortened)
    else:
        raise ValueError("Unknown method: use 'pca' or 'tsne'")
    
    # Generate a color map based on the number of unique labels
    unique_labels = set(labels)
    colors = plt.cm.get_cmap('rainbow', len(unique_labels))
    
    # Plot each cluster with a different color
    for label in unique_labels:
        # Color noise points as black (label == -1)
        if label == -1:
            color = 'k'
        else:
            color = colors(label / len(unique_labels))
        
        plt.scatter(reduced_data[labels == label, 0], reduced_data[labels == label, 1], 
                    c=[color], label=f"Cluster {label}" if label != -1 else "Noise", s=50, edgecolors='k')
    
    plt.title(title)
    plt.legend()
    plt.show()



def DBSCAN_clustering(data):
	# NO ACTUAL CLUSTER CENTERS...
	# DBSCAN clustering
	clustering = DBSCAN(eps=3, min_samples=2).fit(X)
	clmLogger.log(clustering.labels_)  # -1 indicates noise points
	return clustering

def custom_affinitypropagation(data, distance_matrix):
    
    # Convert distances to similarities (AffinityPropagation expects similarities)
    similarity_matrix = -distance_matrix
    
    # Apply Affinity Propagation clustering
    clustering = AffinityPropagation(verbose=True, affinity='precomputed').fit(similarity_matrix)
    
    # Output number of clusters and cluster labels
    clmLogger.log("Number of clusters: {}".format(len(set(clustering.labels_))))
    clmLogger.log("Cluster labels: {}".format(clustering.labels_))
    
    return clustering.labels_, clustering.cluster_centers_indices_, data[clustering.cluster_centers_indices_]



def GMM(data):
    lowest_bic = np.infty
    best_gmm = None
    n_components_range = range(1, 6)
    for n_components in n_components_range:
        gmm = GaussianMixture(n_components=n_components)
        gmm.fit(data)
        bic = gmm.bic(data)
        if bic < lowest_bic:
            lowest_bic = bic
            best_gmm = gmm

    clmLogger.log("Best number of components: {}".format(best_gmm.n_components))
    return best_gmm

def custom_dbscan(X, eps=0.1, min_samples=2, custom_distance_func=None):
    """
    DBSCAN clustering with a custom distance function.

    Parameters:
    - X: The input data (numpy array).
    - eps: The maximum distance between two samples for them to be considered as in the same neighborhood.
    - min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered a core point.
    - custom_distance_func: A custom distance function that takes two points as input and returns their distance.

    Returns:
    - Cluster labels for each point in the dataset.
    """
    if custom_distance_func is None:
        raise ValueError("A custom distance function must be provided.")
    
    # Run DBSCAN with the custom distance function
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=custom_distance_func)
    dbscan.fit(X)
    
    return dbscan.labels_

# Custom distance function (e.g., Manhattan distance)
def custom_manhattan_distance(x, y):
    return np.sum(np.abs(x - y), axis=-1)



def visualize_clusters(data, labels, title="DBSCAN Clustering"):
    """
    Function to visualize clustered data, with each label having a different color.
    
    Parameters:
    - data: numpy array, the dataset
    - labels: numpy array, the cluster labels from DBSCAN
    """
    # Generate a color map based on the number of unique labels
    unique_labels = set(labels)
    colors = plt.cm.get_cmap('rainbow', len(unique_labels))
    
    for label in unique_labels:
        # Color noise points as black (label == -1)
        if label == -1:
            color = 'k'
        else:
            color = colors(label / len(unique_labels))
        
        # Plot each cluster with a different color
        plt.scatter(data[labels == label, 0], data[labels == label, 1], c=[color], label=f"Cluster {label}" if label != -1 else "Noise", s=50, edgecolors='k')
    
    plt.title(title)
    plt.legend()
    plt.show()



def k_medoids(distance_matrix, k, max_iter=100):
    """
    k-medoids clustering algorithm with precomputed distance matrix.
    
    Parameters:
    - distance_matrix: Precomputed distance matrix (n x n).
    - k: Number of clusters (medoids).
    - max_iter: Maximum number of iterations.
    
    Returns:
    - medoids: Final medoids' indices.
    - labels: Cluster labels for each point.
    """
    n = distance_matrix.shape[0]  # Number of data points
    
    # Step 1: Initialize k medoids randomly
    medoids = random.sample(range(n), k)
    
    # Function to assign each point to the nearest medoid
    def assign_labels(medoids):
        labels = np.argmin(distance_matrix[:, medoids], axis=1)
        return labels
    
    # Step 2: Assign each point to the nearest medoid
    labels = assign_labels(medoids)
    
    # Step 3: Iteratively update the medoids
    for _ in range(max_iter):
        new_medoids = medoids.copy()
        
        # Try to update each medoid
        for i in range(k):
            cluster_indices = np.where(labels == i)[0]  # Points assigned to the i-th medoid
            if len(cluster_indices) == 0:
                continue
                
            # Find the point in the cluster with the minimum total distance to all other points in the cluster
            min_total_distance = np.inf
            best_medoid = medoids[i]
            
            for candidate in cluster_indices:
                total_distance = np.sum(distance_matrix[candidate, cluster_indices])
                if total_distance < min_total_distance:
                    min_total_distance = total_distance
                    best_medoid = candidate
            
            new_medoids[i] = best_medoid
        
        # Step 4: Check for convergence (medoids do not change)
        if np.array_equal(medoids, new_medoids):
            break
        
        medoids = new_medoids
        labels = assign_labels(medoids)
    
    return medoids, labels




if __name__=="__main__":
	# Example data - Simple function testing.
	X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])

	X= np.random.rand(100,2)

	clmLogger.log(X[0])
	cluster = DBSCAN_clustering(X)

	cluster2_labels = custom_dbscan(X, 3, 2, custom_manhattan_distance)

	clmLogger.log("Cluster1 labels: {}".format(cluster.labels_))
	clmLogger.log("Cluster2 labels: {}".format(cluster2_labels))


	visualize_clusters(X, cluster2_labels, "Custom DBSCAN")
	visualize_clusters(X, cluster.labels_)

