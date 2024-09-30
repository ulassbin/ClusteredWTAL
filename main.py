import os
import numpy as np
from tslearn.metrics import cdist_dtw
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import random
import time


import clustering_methods as clm
import cluster_analysis as analysis
import helper_functions as helper
import torch_fft

def load_videos(data_dir, num_videos=None):
    video_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')]
    feature_dim = 0
    # Limit the number of videos if specified
    if num_videos is not None and num_videos < len(video_files):
        video_files = random.sample(video_files, num_videos)
        feature_dim = np.load(video_files[0]).shape[1]
    videos = [np.load(file) for file in video_files] # Flatten is necessary for DBSCAN
    lengths = [video.shape[0] for video in videos]
    max_len = max(lengths)
    padded_videos = np.array([np.pad(video, ((0, max_len - video.shape[0]), (0, 0)), 'constant').flatten() for video in videos])
    print("Max len is ", max_len)
    return padded_videos, feature_dim, lengths

# --- Config ---
data_dir = '/home/ulas/Documents/Datasets/CoLA/data/THUMOS14/features/test/rgb'
load_labels = False
cluster_method='affinity_prop'

# Load the videos (specify the number of videos to load, or None to load all)
videos, feature_dim, lengths = load_videos(data_dir, num_videos=100)
helper.feature_dim = feature_dim

plt.plot(lengths)
plt.show()
print("Videos shape {}, {}, features {}".format(len(videos), videos.shape, feature_dim))
cluster_centers = None
if(load_labels):
    labels = np.load('cluster_labels.npy')    
else:
    if(cluster_method == 'DBSCAN'):
        labels = clm.custom_dbscan(videos, eps=0.5*1e-4, min_samples=2, custom_distance_func=helper.fft_distance_2d)
    else: # Distance Precomputed method!
        distance_matrix = helper.cuda_fft_distances(videos, feature_dim, 10) # Second param is batch
        clm.visualize_distance_matrix(distance_matrix)
        labels, cluster_center_indexes, cluster_centers = clm.custom_affinitypropagation(videos, distance_matrix) # We can do better in precomputation hg
    np.save('cluster_labels.npy',labels)

# Visualize the clusters using PCA
clm.visualize_clusters_with_pca(videos, labels, title=cluster_method, method='pca')

if(cluster_centers == None):
    cluster_center_indexes, cluster_centers = clm.getClusterCenters(videos, labels) 

print("From {} videos Cluster indexes are {}".format(len(videos), cluster_center_indexes))