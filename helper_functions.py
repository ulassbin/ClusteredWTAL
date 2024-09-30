import numpy as np
import os
import random
from tslearn.metrics import cdist_dtw
import torch
import numpy as np
import time

feature_dim = 2048

def print_memory_usage(message=""):
    """Print GPU memory usage."""
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    print(f"{message} - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")


#Function to compute cdist_dtw
def comp_cdist_btw(videos):
	return cdist_dtw(videos)

# Function to normalize a video sequence
def normalize(video):
    return (video - np.mean(video)) / np.std(video)


# Function to calculate the FFT-based distance between two videos using 2D FFT
def fft_distance_2d(video1, video2):
    # Normalize the videos
    video1 = np.tanh(video1.reshape(-1,feature_dim)) # old version normalize(video1.reshape(-1,feature_dim))
    video2 = np.tanh(video2.reshape(-1,feature_dim))

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

    convolution_result = np.mean(convolution_result, axis=1) # Get the mean overlapping axis
    # Peak value in the convolution result
    peak_value = np.max(convolution_result)

    # Distance as the inverse of peak value (to ensure similarity yields a small distance)
    distance = 1 / (peak_value + 1e-10)  # Add small value to avoid division by zero
    return distance


def fft_distance_2d_with_pose(video1, video2):
    # Normalize the videos
    video1 = np.tanh(video1.reshape(-1,feature_dim)) # Tanh is quite necessary here!
    video2 = np.tanh(video2.reshape(-1,feature_dim))

    print("Values of 1 feature max {}, mean {}, min {}".format(np.max(video1[0]), np.mean(video1[0]), np.min(video1[0])))

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
    print('Conv result ', convolution_result.shape)
    convolution_result = np.mean(convolution_result, axis=1)
    peak_value = np.max(convolution_result)
    print('Convolution len {}, peak {}, min {}, mean{}'.format(convolution_result.shape, peak_value, np.min(convolution_result), np.mean(convolution_result)))
    index = np.where(peak_value)[0]
    # Distance as the inverse of peak value (to ensure similarity yields a small distance)
    distance = 1 / (peak_value + 1e-10)  # Add small value to avoid division by zero
    return distance, index




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


# Function to normalize a batch of video sequences
def normalize(video_batch):
    return (video_batch - video_batch.mean(dim=(1,2), keepdim=True)) / video_batch.std(dim=(1,2), keepdim=True)

def fft_distance_2d_batch(video_batch1, video_batch2):
    # Normalize the video batches
    video_batch1 = torch.tanh(normalize(video_batch1))
    video_batch2 = torch.tanh(normalize(video_batch2))
    print_memory_usage('After tanh')
    # Get the size of each batch
    n1, d1 = video_batch1.shape[1], video_batch1.shape[2]
    n2, d2 = video_batch2.shape[1], video_batch2.shape[2]

    # Ensure the feature dimension is the same
    assert d1 == d2, "Feature dimensions must match."

    # Determine padding size (sum of temporal dimensions - 1)
    pad_n = n1 + n2 - 1

    # Apply 2D FFT across the entire video (frames, features)
    video_batch1 = torch.fft.fft2(video_batch1, s=(pad_n, d1))
    print_memory_usage('After fft1')
    torch.cuda.empty_cache()
    video_batch2 = torch.fft.fft2(video_batch2, s=(pad_n, d1))
    print_memory_usage('After fft2')
    torch.cuda.empty_cache()

    # Element-wise multiplication in frequency domain and summation over feature dimension
    fft_mult = torch.fft.ifft2(video_batch1 * torch.conj(video_batch2))
    print_memory_usage('After IFFT2')
    print("convolution_result shape before sum", fft_mult.shape)
    convolution_result = torch.real(fft_mult).mean(dim=2)  # Taking mean over feature dimension
    print("convolution_result shape ", convolution_result.shape)
    # Peak value in the convolution result for each pair (across temporal dimension)
    peak_values = torch.amax(convolution_result)

    # Distance as the inverse of peak value (to ensure similarity yields a small distance)
    distances = 1 / (peak_values + 1e-10)  # Add small value to avoid division by zero
    return distances

# Function to compute the distance matrix for a list of videos with batching
def cdist_fft_2d_batched(videos, batch_size=32):
    print("Videos type ", type(videos))
    num_videos = len(videos)
    distance_matrix = torch.zeros((num_videos, num_videos), device=videos[0].device)
    print_memory_usage('After Forming Matrix')
    with torch.no_grad():
        for i in range(num_videos):
            # Prepare batches for parallel processing
            for j in range(i + 1, num_videos, batch_size):
                end = min(j + batch_size, num_videos)
                print_memory_usage('Before forming batches {},{}'.format(i,j))
                batch_videos_j = videos[j:end].to('cuda')  # Pad videos in the batch
                batch_videos_i = videos[i].unsqueeze(0).expand(end - j,  -1, -1).to('cuda')
                print_memory_usage('After Forming batches {},{}'.format(i,j))
                print("Type of batch_videos {} and {}".format(type(batch_videos_i), type(batch_videos_j)))
                print("Shapes of videos {} and {}".format(batch_videos_i.shape, batch_videos_j.shape))
                # Compute distances for the batch
                distances = fft_distance_2d_batch(batch_videos_i, batch_videos_j)

                # Update the distance matrix
                distance_matrix[i, j:end] = distances
                distance_matrix[j:end, i] = distances  # Symmetric matrix
            #del batch_videos_j, batch_videos_i, distances
            torch.cuda.empty_cache()
            print_memory_usage('{} Iteration'.format(i))
    return distance_matrix

def cuda_fft_distances(videos, feature_size, batch=32):
    # Videos shape is, (#num_videos, max_lenxfeature_dim)
    num_videos = len(videos)
    print_memory_usage('Before tensors')
    videos = torch.from_numpy(videos.reshape(num_videos,-1,feature_dim)).float()  # Move videos to GPU
    print_memory_usage('After Converting to Tensor')
    print('Moved video to Gpu, videos type', type(videos))
    distance_matrix = cdist_fft_2d_batched(videos, batch_size=batch)
    print_memory_usage('After Calculating Distance Matrix')
    time.sleep(2)
    torch.cuda.empty_cache()
    print_memory_usage('Emptied cache')
    return distance_matrix.to('cpu').numpy()
