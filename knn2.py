import os
import numpy as np
from tslearn.metrics import cdist_dtw
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import random

import time

import cluster_analysis as analysis
import helper_functions as func
import torch_fft


def load_videos(data_dir, num_videos=None):
    video_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')]
    
    # Limit the number of videos if specified
    if num_videos is not None and num_videos < len(video_files):
        video_files = random.sample(video_files, num_videos)
    
    videos = [np.load(file) for file in video_files]
    return videos

# Directory containing the .npy files
data_dir = '/home/ulas/Documents/Datasets/CoLA/data/THUMOS14/features/test/rgb/'

# Load the videos (specify the number of videos to load, or None to load all)
videos = load_videos(data_dir, num_videos=200)

# Compute the pairwise DTW distance matrix using tslearn
start_time = time.time()

#distance_matrix = func.comp_cdist_btw(videos)
distance_matrix = torch_fft.cuda_fft_distances(data_dir, 100)
#distance_matrix = func.comp_cdist_fft_2d(videos)
end_time = time.time()
duration = end_time - start_time
print(f"Distance Calculation took {duration:.2f} seconds to execute.")
#start_time = time.time()
analysis.elbow_and_silhoutte(distance_matrix, 10)
#end_time = time.time()
#duration = end_time - start_time
#print(f"Cluster Calculation took {duration:.2f} seconds to execute.")
# Perform K-means clustering using the distance matrix
#kmeans = KMeans(n_clusters=4, random_state=0)
#labels = kmeans.fit_predict(distance_matrix)

#silhouette = analysis.average_silhouette_scores_per_cluster(distance_matrix, labels)
#analysis.plot_silhouette_scores(silhouette)

# Use MDS to reduce the dimensionality of the distance matrix to 2D for visualization
#mds = MDS(n_components=2, dissimilarity="precomputed", random_state=0)
#embedding = mds.fit_transform(distance_matrix)

# Visualize the clusters
#plt.figure(figsize=(10, 7))
#for label in np.unique(labels):
#    plt.scatter(embedding[labels == label, 0], embedding[labels == label, 1], label=f"Cluster {label + 1}")

#plt.title("K-means Clustering of Video Sequences")
#plt.xlabel("MDS Dimension 1")
#plt.ylabel("MDS Dimension 2")
#plt.legend()
#plt.show()
