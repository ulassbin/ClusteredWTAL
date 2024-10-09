import os
import numpy as np
from logger import logging, CustomLogger

logger = CustomLogger('DataLoader')
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
    logger.log("Max len is {}".format(max_len))
    return padded_videos, feature_dim, lengths, max_len

def load_distance_matrix():
	pass

def load_cluster_information(base_path):
    labels = cluster_centers = cluster_center_indexes = None
    print('{}/cluster_labels.npy'.format(base_path))
    if (os.path.exists('{}/cluster_labels.npy'.format(base_path))):
        labels = np.load('{}/cluster_labels.npy'.format(base_path))
    if (os.path.exists('{}/cluster_centers.npy'.format(base_path))):    
        cluster_centers = np.load('{}/cluster_centers.npy'.format(base_path))
    if(os.path.exists('{}/cluster_center_indexes.npy'.format(base_path))):    
        cluster_center_indexes = np.load('{}/cluster_center_indexes.npy'.format(base_path))
    return labels, cluster_centers, cluster_center_indexes