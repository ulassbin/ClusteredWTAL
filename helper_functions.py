import numpy as np
import os
import random
from tslearn.metrics import cdist_dtw

#Function to compute cdist_dtw
def comp_cdist_btw(videos):
	return cdist_dtw(videos)

# Function to normalize a video sequence
def normalize(video):
    return (video - np.mean(video)) / np.std(video)

# Function to calculate the FFT-based distance between two videos using 2D FFT
def fft_distance_2d(video1, video2):
    # Normalize the videos
    video1 = normalize(video1)
    video2 = normalize(video2)

    # Get the size of each video
    n1, d1 = video1.shape
    n2, d2 = video2.shape

    # Determine padding size (maximum of dimensions)
    pad_n = n1 + n2 - 1
    pad_d = max(d1, d2)

    # Apply 2D FFT with padding to the next largest size
    fft_video1 = np.fft.fft2(video1, s=(pad_n, pad_d))
    fft_video2 = np.fft.fft2(video2, s=(pad_n, pad_d))

    # Element-wise multiplication in frequency domain
    fft_mult = fft_video1 * np.conj(fft_video2)

    # Compute the inverse 2D FFT to get the convolution result
    convolution_result = np.fft.ifft2(fft_mult)
    convolution_result = np.real(convolution_result)

    # Peak value in the convolution result
    peak_value = np.max(convolution_result)

    # Distance as the inverse of peak value (to ensure similarity yields a small distance)
    distance = 1 / (peak_value + 1e-10)  # Add small value to avoid division by zero
    return distance

# Function to compute the distance matrix for a list of videos
def comp_cdist_fft_2d(videos):
    num_videos = len(videos)
    distance_matrix = np.zeros((num_videos, num_videos))
    
    for i in range(num_videos):
        for j in range(i + 1, num_videos):
            distance_matrix[i, j] = fft_distance_2d(videos[i], videos[j])
            distance_matrix[j, i] = distance_matrix[i, j]  # Symmetric matrix
    
    return distance_matrix
