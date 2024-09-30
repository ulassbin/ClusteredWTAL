from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def print_memory_usage(message=""):
    """Print GPU memory usage."""
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    print(f"{message} - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")


def getClusterCenters(data, labels, custom_distance):
    unique_labels = np.unique(labels)
    print("unique_labels ", unique_labels)
    medoid_indexes = []
    for label in unique_labels:
        print("------LABEL ",label)
        cluster_indexes = np.where(labels == label)[0]
        cluster_data = data[cluster_indexes]
        num_points = cluster_data.shape[0]
        # Calculate distance matrix (upper triangle since direction doesn't matter)
        distance_matrix = np.zeros((num_points, num_points))
        for i in range(num_points):
            for j in range(i+1, num_points):
                distance = custom_distance(cluster_data[i], cluster_data[j])
                print('Distance between {} and {} is {}'.format(i, j, distance))
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance  # Symmetric matrix
        total_distances = np.sum(distance_matrix, axis=1)
        medoid_index = np.argmin(total_distances)
        print("medoid_index", medoid_index)
        final_index = cluster_indexes[medoid_index]
        medoid_indexes.append(final_index)

    medoid_indexes = np.array(medoid_indexes)
    print('Medoid indexes ', medoid_indexes)
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
    flattened_data = data.reshape(data.shape[0], -1)
    
    # Apply dimensionality reduction (PCA or t-SNE)
    if method == 'pca':
        reducer = PCA(n_components=2)  # Reduce to 2 dimensions
        reduced_data = reducer.fit_transform(flattened_data)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=0)
        reduced_data = reducer.fit_transform(flattened_data)
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
	print(clustering.labels_)  # -1 indicates noise points
	return clustering

def custom_affinitypropagation(data, distance_matrix):
    
    # Convert distances to similarities (AffinityPropagation expects similarities)
    similarity_matrix = -distance_matrix
    
    # Apply Affinity Propagation clustering
    clustering = AffinityPropagation(verbose=True, affinity='precomputed').fit(similarity_matrix)
    
    # Output number of clusters and cluster labels
    print("Number of clusters:", len(set(clustering.labels_)))
    print("Cluster labels:", clustering.labels_)
    
    return clustering.labels_, clustering.cluster_centers_indices_, data[clustering.cluster_centers_indices_]



def GMM(data):
	lowest_bic = np.infty
	best_gmm = None
	n_components_range = range(1, 6)
	for n_components in n_components_range:
	    gmm = GaussianMixture(n_components=n_components)
	    gmm.fit(X)
	    bic = gmm.bic(X)
	    if bic < lowest_bic:
	        lowest_bic = bic
	        best_gmm = gmm

	print("Best number of components:", best_gmm.n_components)
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



if __name__=="__main__":
	# Example data - Simple function testing.
	X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])

	X= np.random.rand(100,2)

	print(X[0])
	cluster = DBSCAN_clustering(X)

	cluster2_labels = custom_dbscan(X, 3, 2, custom_manhattan_distance)

	print("Cluster1 labels:", cluster.labels_)
	print("Cluster2 labels:", cluster2_labels)


	visualize_clusters(X, cluster2_labels, "Custom DBSCAN")
	visualize_clusters(X, cluster.labels_)

