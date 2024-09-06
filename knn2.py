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
data_dir = '/abyss/shared/users/ulas-bingoel/THUMOS14/features/test/rgb'

# Load the videos (specify the number of videos to load, or None to load all)
videos = load_videos(data_dir, num_videos=200)

# Compute the pairwise DTW distance matrix using tslearn
start_time = time.time()

#distance_matrix = func.comp_cdist_btw(videos)
distance_matrix = torch_fft.cuda_fft_distances(data_dir, 100, 100)
#distance_matrix = func.comp_cdist_fft_2d(videos)
#distance_matrix = 0
end_time = time.time()
duration = end_time - start_time
print(f"Distance Calculation took {duration:.2f} seconds to execute.")
analysis.elbow_and_silhoutte(distance_matrix, 10)
