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
import data_loader as loader
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


def visualize_clustering_heatmap(distance_matrix, labels, title="Clustering Heatmap", save_dir='figures/heatmap.png'):
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
    # how to save these plots
    plt.savefig(save_dir)





# --- Config --- /abyss/home/THUMOS14/features/train/rgb
data_dirs = ['/abyss/home/THUMOS14/features/train/rgb']
#data_dirs = ['/abyss/home/THUMOS14/features/train/rgb', '/abyss/home/THUMOS14/features/test/rgb']
#data_dir = '/home/ulas/Documents/Datasets/CoLA/data/THUMOS14/features/train/rgb'
cluster_dir = './data'
load_labels = False
cluster_method='affinity'
num_videos = None
distance_comp_batch = 30 # 30
quick_run = True
# Load the videos (specify the number of videos to load, or None to load all)
simple_loader = loader.simpleLoader(data_dirs, cluster_dir, quick_run=quick_run)
video_files, feature_dim, lengths, max_len = simple_loader.load_videos()
helper.feature_dim = clm.feature_dim = feature_dim

plt.plot(lengths)
plt.show()
#mainLogger.log("Videos shape {}, {}, features {}".format(len(videos), videos.shape, feature_dim), logging.WARNING)
cluster_centers = None
if(load_labels):
    labels, cluster_centers, cluster_center_indexes = loader.load_cluster_information()
else:
    if(cluster_method == 'DBSCAN'):
        labels = clm.custom_dbscan(videos, eps=0.5*1e-4, min_samples=2, custom_distance_func=helper.fft_distance_2d)
    elif(cluster_method == 'kmedoids'):
        distance_matrix = helper.cuda_fft_distances(video_files, simple_loader, feature_dim, distance_comp_batch) # Second param is batch
        clm.visualize_distance_matrix(distance_matrix)
        exit()
        cluster_center_indexes, labels = clm.k_medoids(distance_matrix, k=10, max_iter = 200) 
        #visualize_distance_heatmap(distance_matrix, labels, title="Distance Matrix Heatmap")
        visualize_clustering_heatmap(distance_matrix, labels, title="Clustering Heatmap")
        center_files = video_files[cluster_center_indexes]
        cluster_centers = simple_loader.load_mini_batch(center_files)
    else: # Distance Precomputed method!
        distance_matrix = helper.cuda_fft_distances(video_files, simple_loader, feature_dim, distance_comp_batch) # Second param is batch
        clm.visualize_distance_matrix(distance_matrix)
        labels, cluster_center_indexes, center_files = clm.custom_affinitypropagation(video_files, distance_matrix) # We can do better in precomputation hg
        cluster_centers = simple_loader.load_mini_batch(center_files)
    np.save('data/cluster_labels.npy',labels)


if(cluster_centers is None or len(cluster_centers) == 0):
    cluster_center_indexes, cluster_centers = clm.getClusterCenters(videos, labels, distance_comp_batch,  helper.fft_distance_2d_batch) 

np.save('data/cluster_centers.npy', cluster_centers)
np.save('data/cluster_center_indexes.npy', cluster_center_indexes)

mainLogger.log("From {} videos Cluster indexes are {}".format(len(videos), cluster_center_indexes))

try:
    videos = simple_loader.load_mini_batch(video_files)
    # Visualize the clusters using PCA
    clm.visualize_clusters_with_pca(videos, labels, title=cluster_method, method='tsne')
except:
    print("PCA Visualization failed")