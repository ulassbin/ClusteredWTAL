import torch
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import helper_functions as helper
import data_loader as loader

print('Ran!')


dist_func = helper.fft_distance_2d
dist_func_batch = helper.fft_distance_2d_batch

# Config ---
data_dir = '/home/ulas/Documents/Datasets/CoLA/data/THUMOS14/features/test/rgb'

videos, feature_dim, length = loader.load_videos(data_dir)

print('Videos shape ', videos.shape)