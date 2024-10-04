import os
import numpy as np
from tslearn.metrics import cdist_dtw
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import random
import time
import seaborn as sns

from logger import logging, CustomLogger
import clustering_methods as clm
import cluster_analysis as analysis
import helper_functions as helper
import torch_fft

mainLogger = CustomLogger(name='main')


def visualize_distance_heatmap(distance_matrix, labels, title="Distance Matrix Heatmap"):
    """
    Visualize the distance matrix as a heatmap with annotations based on labels.

    Parameters:
    - distance_matrix: Precomputed distance matrix (n x n).
    - labels: Labels for each data point (for annotation).
    - title: Title of the heatmap.
    """
    # Ensure distance_matrix is a NumPy array for consistency
    distance_matrix = np.array(distance_matrix)
    
    # Create the heatmap using seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(distance_matrix, annot=False, cmap='coolwarm', linewidths=0.5)

    # Add title and labels
    plt.title(title)
    plt.xlabel('Data Points')
    plt.ylabel('Data Points')
    
    # Optionally, label the axes with the computed labels
    unique_labels = np.unique(labels)
    plt.xticks(ticks=np.arange(len(labels)) + 0.5, labels=labels, rotation=90)
    plt.yticks(ticks=np.arange(len(labels)) + 0.5, labels=labels, rotation=0)
    
    # Show the heatmap
    plt.tight_layout()
    plt.show()


def visualize_clustering_heatmap(distance_matrix, labels, title="Clustering Heatmap"):
    """
    Visualize clustering assignments using a heatmap of the distance matrix.

    Parameters:
    - distance_matrix: Precomputed distance matrix (n x n).
    - labels: Cluster labels for each data point.
    - title: Title of the heatmap.
    """
    # Ensure distance_matrix is a NumPy array
    distance_matrix = np.array(distance_matrix)
    
    # Sort distance matrix based on cluster labels
    sorted_indices = np.argsort(labels)
    sorted_distances = distance_matrix[sorted_indices, :][:, sorted_indices]
    sorted_labels = np.array(labels)[sorted_indices]

    # Create a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(sorted_distances, annot=False, cmap='coolwarm', linewidths=0.5, cbar=True)
    
    # Add title and labels
    plt.title(title)
    plt.xlabel('Data Points (sorted by cluster)')
    plt.ylabel('Data Points (sorted by cluster)')

    # Optionally add cluster label ticks
    plt.xticks(ticks=np.arange(len(sorted_labels)), labels=sorted_labels, rotation=90)
    plt.yticks(ticks=np.arange(len(sorted_labels)), labels=sorted_labels, rotation=0)
    
    plt.tight_layout()
    plt.show()




def load_videos(data_dir, num_videos=None):
    video_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')]
    feature_dim = np.load(video_files[0]).shape[1]
    # Limit the number of videos if specified
    if num_videos is not None and num_videos < len(video_files):
        video_files = random.sample(video_files, num_videos)
    videos = [np.load(file) for file in video_files] # Flatten is necessary for DBSCAN
    lengths = [video.shape[0] for video in videos]
    max_len = max(lengths)
    padded_videos = np.array([np.pad(video, ((0, max_len - video.shape[0]), (0, 0)), 'constant').flatten() for video in videos])
    mainLogger.log("Max len is {}".format(max_len))
    return padded_videos, feature_dim, lengths

# --- Config ---
data_dir = '/home/ulas/Documents/Datasets/CoLA/data/THUMOS14/features/test/rgb'
load_labels = True
cluster_method='affinity'
num_videos = None
distance_comp_batch = 30 # 30
# Load the videos (specify the number of videos to load, or None to load all)
videos, feature_dim, lengths = load_videos(data_dir, num_videos)
helper.feature_dim = clm.feature_dim = feature_dim

plt.plot(lengths)
plt.show()
mainLogger.log("Videos shape {}, {}, features {}".format(len(videos), videos.shape, feature_dim), logging.WARNING)
cluster_centers = None
if(load_labels):
    if (os.path.exists('cluster_labels.npy')):
        labels = np.load('cluster_labels.npy')
    if (os.path.exists('cluster_centers.npy')):    
        cluster_centers = np.load('cluster_centers')
    if(os.path.exists('cluster_center_indexes.npy')):    
        cluster_centers = np.load('cluster_center_indexes.npy')
else:
    if(cluster_method == 'DBSCAN'):
        labels = clm.custom_dbscan(videos, eps=0.5*1e-4, min_samples=2, custom_distance_func=helper.fft_distance_2d)
    elif(cluster_method == 'kmedoids'):
        distance_matrix = helper.cuda_fft_distances(videos, feature_dim, distance_comp_batch) # Second param is batch
        clm.visualize_distance_matrix(distance_matrix)
        cluster_center_indexes, labels = clm.k_medoids(distance_matrix, k=10, max_iter = 200) 
        #visualize_distance_heatmap(distance_matrix, labels, title="Distance Matrix Heatmap")
        visualize_clustering_heatmap(distance_matrix, labels, title="Clustering Heatmap")
        cluster_centers = videos[cluster_center_indexes]
    else: # Distance Precomputed method!
        distance_matrix = helper.cuda_fft_distances(videos, feature_dim, distance_comp_batch) # Second param is batch
        clm.visualize_distance_matrix(distance_matrix)
        labels, cluster_center_indexes, cluster_centers = clm.custom_affinitypropagation(videos, distance_matrix) # We can do better in precomputation hg
    np.save('cluster_labels.npy',labels)

# Visualize the clusters using PCA
clm.visualize_clusters_with_pca(videos, labels, title=cluster_method, method='tsne')

if(cluster_centers == None):
    cluster_center_indexes, cluster_centers = clm.getClusterCenters(videos, labels, distance_comp_batch,  helper.fft_distance_2d_batch) 
    np.save('cluster_centers.npy', cluster_centers)
    np.save('cluster_center_indexes.npy', cluster_center_indexes)

mainLogger.log("From {} videos Cluster indexes are {}".format(len(videos), cluster_center_indexes))