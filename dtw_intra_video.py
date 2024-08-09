import numpy as np
from tslearn.metrics import cdist_dtw
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import cluster_analysis as analysis
# Generate some example video sequences with random lengths and 1024 feature dimensions
videos = [np.random.rand(np.random.randint(80, 120), 1024) for _ in range(100)]

# Compute the pairwise DTW distance matrix using tslearn
distance_matrix = cdist_dtw(videos)

# Perform K-means clustering using the distance matrix
kmeans = KMeans(n_clusters=5, random_state=20)
labels = kmeans.fit_predict(distance_matrix)

silhouette = analysis.average_silhouette_scores_per_cluster(distance_matrix, labels)
analysis.plot_silhouette_scores(silhouette)

# Use MDS to reduce the dimensionality of the distance matrix to 2D for visualization
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=0)
embedding = mds.fit_transform(distance_matrix)

# Visualize the clusters
plt.figure(figsize=(10, 7))
for label in np.unique(labels):
    plt.scatter(embedding[labels == label, 0], embedding[labels == label, 1], label=f"Cluster {label + 1}")

plt.title("K-means Clustering of Video Sequences")
plt.xlabel("MDS Dimension 1")
plt.ylabel("MDS Dimension 2")
plt.legend()
plt.show()
