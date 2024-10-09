import numpy as np
import os
import random
from tslearn.metrics import cdist_dtw
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math
import copy

from logger import logging, CustomLogger

helperLogger = CustomLogger('helper')
full_conv = False

feature_dim = 2048

def print_memory_usage(message=""):
    """Print GPU memory usage."""
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    helperLogger.log(f"{message} - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")


#Function to compute cdist_dtw
def comp_cdist_btw(videos):
	return cdist_dtw(videos)

# Function to normalize a video sequence
def normalize(video):
    return (video - np.mean(video)) / np.std(video)


# Function to calculate the FFT-based distance between two videos using 2D FFT
def fft_distance_2d(video1, video2):
    # Normalize the videos
    video1 = torch.tensor(video1.reshape(-1, feature_dim))
    video2 = torch.tensor(video2.reshape(-1, feature_dim))
    video1 = F.normalize(video1, dim=1) # old version normalize(video1.reshape(-1,feature_dim))
    video2 = F.normalize(video2, dim=1)

    n1, d1 = video1.shape[0], video1.shape[1]
    n2, d2 = video2.shape[0], video2.shape[1]

    # Ensure the feature dimension is the same
    assert d1 == d2, "Feature dimensions must match."

    # Determine padding size (sum of temporal dimensions - 1)
    pad_n = n1  # This is the padding size for the temporal dimension

    video1_fft = torch.fft.fft2(video1, dim=0)
    #torch.cuda.empty_cache()
    
    video2_fft = torch.fft.fft2(video2, dim=0)
    #torch.cuda.empty_cache()

    # Element-wise multiplication in frequency domain and summation over feature dimension
    fft_mult = torch.fft.ifft2(video1_fft * torch.conj(video2_fft), dim=0)
    convolution_result = torch.real(fft_mult).sum(dim=1)  # Sum over feature dimension


    #print("convolution_result shape ", convolution_result.shape)
    # Peak value in the convolution result for each pair (across temporal dimension)
    if(not full_conv):
        peak_values = torch.amax(convolution_result)
        # Distance as the inverse of peak value (to ensure similarity yields a small distance)
        distances = 1 / (peak_values + 1e-10)  # Add small value to avoid division by zero
    else:
        distances = 1 / (convolution_result + 1e-10)  # Add small value to avoid division by zero    
    return distances


# Function to compute the distance matrix for a list of videos
def comp_cdist_fft_2d(videos):
    num_videos = len(videos)
    distance_matrix = np.zeros((num_videos, num_videos))
    
    for i in range(num_videos):
        for j in range(i + 1, num_videos):
            distance_matrix[i, j] = fft_distance_2d(videos[i], videos[j])
            distance_matrix[j, i] = distance_matrix[i, j]  # Symmetric matrix
    
    return distance_matrix



# --- Batched computations below


def circular_cosine_similarity(video_batch1, video_batch2):
    """
    Computes the cosine similarity between two video batches at all possible circular shifts.
    
    Parameters:
    - video_batch1: Tensor of shape [batch, time, feature]
    - video_batch2: Tensor of shape [batch, time, feature], shifted version of video_batch1
    
    Returns:
    - result: Tensor of shape [batch, time], containing the summed cosine similarity for each circular shift.
    """
    # Ensure both batches have the same shape
    assert video_batch1.shape == video_batch2.shape, "Both video batches must have the same shape"
    
    batch_size, time_dim, feature_dim = video_batch1.shape
    
    # Normalize both video batches along the feature dimension (cosine similarity normalization)
    video_batch1_norm = F.normalize(video_batch1, dim=2)
    video_batch2_norm = F.normalize(video_batch2, dim=2)

    # Initialize a tensor to store the cosine similarity results for all shifts
    result = torch.zeros((batch_size, time_dim))  # [batch, time] for circular shifts
    
    # Compute cosine similarity at each circular shift
    for shift in range(time_dim):  # Shift range from 0 to time_dim-1
        # Circular shift video_batch2 by the current shift amount
        shifted_video_batch2 = torch.roll(video_batch2_norm, shifts=shift, dims=1)

        # Compute cosine similarity over the feature dimension
        cosine_sim = torch.sum(video_batch1_norm * shifted_video_batch2, dim=2)  # [batch, time]

        # Sum over all time steps for the current shift
        result[:, shift] = cosine_sim.sum(dim=1)  # Summing over time steps

    return result

def plot_comparison(convolution_result, shifted_sum_result):
    num_plots = convolution_result.shape[0]  # Number of subplots (same as the batch size)

    # Determine the number of rows and columns to make the plot layout roughly square
    num_cols = math.ceil(math.sqrt(num_plots))  # Number of columns
    num_rows = math.ceil(num_plots / num_cols)  # Number of rows

    # Set up the figure and adaptive subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))  # Adjust the size as needed

    # Flatten axes in case of multi-dimensional axes array for easier access
    axes = axes.flatten()

    # Loop over each batch index and plot corresponding FFT and shifted sum results in a separate subplot
    for i in range(num_plots):
        ax = axes[i]  # Get the corresponding axis for the current plot

        # Plot FFT result
        ax.plot(convolution_result[i].cpu().numpy(), label=f'FFT Result {i+1}', color='b')

        # Plot Shifted Sum result
        ax.plot(shifted_sum_result[i].cpu().numpy(), label=f'ShiftedSum Result {i+1}', color='r', linestyle='--')

        # Set labels and title
        ax.set_xlabel('X-axis (Time)')
        ax.set_ylabel('Y-axis (Values)')
        ax.set_title(f'Comparison for Index {i+1}')

        # Add legend for each subplot
        ax.legend()

    # Remove any unused axes (in case num_plots isn't a perfect square)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout to avoid overlapping subplots
    plt.tight_layout()

    # Display the plot
    plt.show()



# Function to normalize a batch of video sequences
def normalize(video_batch):
    return (video_batch - video_batch.mean(dim=(1,2), keepdim=True)) / video_batch.std(dim=(1,2), keepdim=True)

def fft_distance_2d_batch(video_batch1, video_batch2):
    # Normalize the video batches
    video_batch1 = F.normalize(video_batch1, dim=2)
    video_batch2 = F.normalize(video_batch2, dim=2)
    #video_batch2 = torch.roll(video_batch1, shifts=100, dims=1)  # Circular shift along temporal dimension (dim=1) to test

    # Get the size of each batch
    n1, d1 = video_batch1.shape[1], video_batch1.shape[2]
    n2, d2 = video_batch2.shape[1], video_batch2.shape[2]

    # Ensure the feature dimension is the same
    assert d1 == d2, "Feature dimensions must match."

    # Determine padding size (sum of temporal dimensions - 1)
    pad_n = n1  # This is the padding size for the temporal dimension

    # Apply 2D FFT across the entire video batch (frames, features)
    # Apply padding in the temporal dimension (pad_n) but keep feature dimension (d1) the same
    video_batch1_fft = torch.fft.fft2(video_batch1, dim=1)
    #torch.cuda.empty_cache()
    
    video_batch2_fft = torch.fft.fft2(video_batch2, dim=1)
    #torch.cuda.empty_cache()

    # Element-wise multiplication in frequency domain and summation over feature dimension
    fft_mult = torch.fft.ifft2(video_batch1_fft * torch.conj(video_batch2_fft), dim=1)
    convolution_result = torch.real(fft_mult).sum(dim=2)  # Sum over feature dimension

    # Peak value in the convolution result for each pair (across temporal dimension)
    #helperLogger.log("Data shape {}".format(video_batch1.shape))
    #helperLogger.log("Convolution result shape  {}".format(convolution_result.shape))

    # for i in range(convolution_result.shape[0]):
    #     plt.plot(convolution_result[i].cpu().numpy(), label=f'FFT Result {i+1}')
    # # Add labels and title
    # plt.xlabel('X-axis (Time)')
    # plt.ylabel('Y-axis (Values)')
    # plt.title('Convolution with a single Data')
    
    # # Optionally add a legend
    # plt.legend()
    
    # # Display the plot
    # plt.tight_layout()
    # plt.show()


    if(not full_conv):
        peak_values = torch.amax(convolution_result)
        # Distance as the inverse of peak value (to ensure similarity yields a small distance)
        distances = 1 / (peak_values + 1e-10)  # Add small value to avoid division by zero
    else:
        distances = 1 / (convolution_result + 1e-10)  # Add small value to avoid division by zero    
    return distances

# Function to compute the distance matrix for a list of videos with batching
def cdist_fft_2d_batched(videos, batch_size=32):
    num_videos = len(videos)
    distance_matrix = torch.zeros((num_videos, num_videos), device=videos[0].device)
    #print_memory_usage('After Forming Matrix')
    prev = copy.deepcopy(full_conv)
    full_conv = False
    with torch.no_grad():
        for i in range(num_videos):
            # Prepare batches for parallel processing
            for j in range(i + 1, num_videos, batch_size):
                end = min(j + batch_size, num_videos)
                #print_memory_usage('Before forming batches {},{}'.format(i,j))
                batch_videos_j = videos[j:end].to('cuda')  # Pad videos in the batch
                batch_videos_i = videos[i].unsqueeze(0).expand(end - j,  -1, -1).to('cuda')
                #print_memory_usage('After Forming batches {},{}'.format(i,j))
                #print("Type of batch_videos {} and {}".format(type(batch_videos_i), type(batch_videos_j)))
                #print("Shapes of videos {} and {}".format(batch_videos_i.shape, batch_videos_j.shape))
                # Compute distances for the batch
                distances = fft_distance_2d_batch(batch_videos_i, batch_videos_j)

                # Update the distance matrix
                distance_matrix[i, j:end] = distances
                distance_matrix[j:end, i] = distances  # Symmetric matrix
            #del batch_videos_j, batch_videos_i, distances
            torch.cuda.empty_cache()
            print_memory_usage('{} Iteration'.format(i))
    full_conv = prev
    return distance_matrix

def cuda_fft_distances(videos, feature_size, batch=32):
    # Videos shape is, (#num_videos, max_lenxfeature_dim)
    helperLogger.log('Videos shape here {} {}'.format(len(videos), videos.shape))
    num_videos = len(videos)
    print_memory_usage('Before tensors')
    videos = torch.from_numpy(videos.reshape(num_videos,-1,feature_dim)).float()  # Move videos to GPU
    print_memory_usage('After Converting to Tensor')
    helperLogger.log('Moved video to Gpu, videos type', type(videos))
    distance_matrix = cdist_fft_2d_batched(videos, batch_size=batch)
    print_memory_usage('After Calculating Distance Matrix')
    time.sleep(2)
    torch.cuda.empty_cache()
    print_memory_usage('Emptied cache')
    return distance_matrix.to('cpu').numpy()
