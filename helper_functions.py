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
def fft_distance_2d(video1, video2, full_conv = True):
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

def fft_distance_2d_batch(video_batch1, video_batch2, full_conv=True):
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
def cdist_fft_2d_batched(videos, batch_size=32, full_conv =False):
    num_videos = len(videos)
    distance_matrix = torch.zeros((num_videos, num_videos), device=videos[0].device)
    #print_memory_usage('After Forming Matrix')
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
                distances = fft_distance_2d_batch(batch_videos_i, batch_videos_j, full_conv)

                # Update the distance matrix
                distance_matrix[i, j:end] = distances
                distance_matrix[j:end, i] = distances  # Symmetric matrix
            #del batch_videos_j, batch_videos_i, distances
            torch.cuda.empty_cache()
            print_memory_usage('{} Iteration'.format(i))
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



def compute_cluster_distances(x, cluster_centers):
    batch_size, temporal_length, feature_dim = x.shape  # (batch_size, temporal_length, feature_dim)
    centers, cent_length, cent_feature_dim = cluster_centers.shape  # (centers, temporal_length, feature_dim)

    # Ensure that the temporal and feature dimensions of x and cluster centers match
    #print('Temporal length {} , centroid length {}'.format(temporal_length, cent_length))
    assert temporal_length == cent_length, "Temporal lengths must match between x and cluster centers."
    assert feature_dim == cent_feature_dim, "Feature dimensions must match between x and cluster centers."

    # Expand x to (batch_size * centers, temporal_length, feature_dim)
    x_expanded = x.unsqueeze(1).expand(batch_size, centers, temporal_length, feature_dim).reshape(-1, temporal_length, feature_dim)

    # Expand cluster_centers to (batch_size * centers, temporal_length, feature_dim)
    cluster_centers_expanded = cluster_centers.unsqueeze(0).expand(batch_size, centers, temporal_length, feature_dim).reshape(-1, temporal_length, feature_dim)

    # Compute distances using fft_distance_2d_batch; the result will be (batch_size * centers, temporal_length)
    distances_expanded = fft_distance_2d_batch(x_expanded, cluster_centers_expanded)  # Shape: (batch_size * centers, temporal_length)

    # Reshape the distances to (batch_size, centers, temporal_length) and then transpose to (batch_size, temporal_length, centers)
    distances = distances_expanded.view(batch_size, centers, temporal_length).permute(0, 2, 1).to('cuda')  # Shape: (batch_size, temporal_length, centers)

    return distances


# Given the cas_to_proposals method below. We need to define a new function to nms the proposals
# Lets implement our own method with a simple approach.
# We can sort the proposals based on the normalized score and then iterate through them
# If the current proposal has an overlap with any of the previous proposals, we can ignore it
# Otherwise we can add it to the final list

# what should be the nms_threshodl values here for example
# Lets say we have 2 proposals
# Proposal 1: [0.2, 1, 1, 0.8, 0.8]
# Proposal 2: [0.0, 0.5, 1.5, 0.7, 0.7]


def nms(proposals, nms_threshold):
    # Sort the proposals based on the normalized score
    proposals = sorted(proposals, key=lambda x: x[4], reverse=True)
    final_proposals = []
    for i in range(len(proposals)):
        current_proposal = proposals[i]
        if len(final_proposals) == 0:
            final_proposals.append(current_proposal)
        else:
            # Check for overlap with the previous proposals
            overlap = False
            for j in range(len(final_proposals)):
                # Calculate intersection over union (IoU)
                start = max(current_proposal[1], final_proposals[j][1])
                end = min(current_proposal[2], final_proposals[j][2])
                intersection = max(0, end - start)
                union = (current_proposal[2] - current_proposal[1]) + (final_proposals[j][2] - final_proposals[j][1]) - intersection
                iou = intersection / union
                if iou > nms_threshold:
                    overlap = True
                    break
            if not overlap:
                final_proposals.append(current_proposal)
    return final_proposals

# Isnt cas enough to get proposals? why do I need actionness?
# I think actionness is used to filter out the proposals
# how do I use actionness to filter out proposals?
def actionness_filter_proposals(proposals, actionness, cfg):
    # Lets say actionness is of shape (temporal_length, num_classes)
    # We can filter out proposals that have actionness less than a threshold
    # Lets say we have a threshold of 0.5
    # proposals shape are (num_classes, start, end, score, normalized_score)
    seconds_to_index = cfg.FEATS_FPS / 16
    threshold = cfg.ANESS_THRESH
    filtered_proposals = []
    for proposal in proposals:
        start_frame = int(proposal[1] * seconds_to_index)
        end_frame = int(proposal[2] * seconds_to_index)
        actionness_values = actionness[start_frame:end_frame]
        if np.mean(actionness_values) > threshold:
            filtered_proposals.append(proposal)
    return filtered_proposals

def cas_to_proposals(cas, threshold, min_proposal_length, fps):
    # Cas is (temporal_length, num_classes)
    # Lets convert cas to 0 and 1 array
    # Lets get the time factor
    num_classes = cas.shape[1]
    index_to_seconds = 16 / fps
    # Time index i corresponds to t=i/t_factor seconds
    cas_thresh = cas > threshold
    # now somehow we should convert cas_thresh into a proposal list of form [class, start, end, score, normalized_score]
    # Lets start with a simple approach
    proposals = [[] for _ in range(self.num_classes)] 
    for i in range(num_classes):
        class_cas = cas_thresh[:,i]
        for j in range(class_cas.shape[0]):
            start = -1
            end = -1
            score = 0
            if class_cas[j] == 1:
                if start == -1:
                    start = j
                end = j
                score += cas[j,i]
            else:
                if start != -1:
                    if end - start > min_proposal_length:
                        current_length = end - start
                        normalized_score = score / current_length
                        proposals[i].append([i, start*index_to_seconds, end*index_to_seconds, score, normalized_score]) 
                    start = -1
                    end = -1
                    score = 0
    return proposals

# lets given two proposals, calculate the IoU and average IoU. For thresholds use cfg.TIOU_THRESH
# proposals are in the form of proposal[num_class] = [class, start, end, score, normalized_score]
# Lets calculate the IoU for each proposal pair

# We need to calculate Iou accuracies for each class and then average them
def calculate_IoU(proposal1, proposal_label):
    # First check if the proposals are of the same class
    # If they are not of the same class, return 0
    match_indexes = []
    for classf in range(len(proposal1)):
        if proposal1[classf][0] != proposal_label[classf][0]:
            return 0 
        match_indexes.append(classf)
    # Now just iterate over the matched indexes and calculate the IoU
    correspondences = [] # To store class proposal to label prooposal matches by index and IOU
    # final matches elements should be something like [[proposal],[label],iou]
    current_iou = 0
    final_average_iou = 0
    for index in match_indexes: # 
        class_proposal1 = proposal1[index]
        class_label = proposal_label[index]
        # how to match the proposals to label proposals?
        # Lets say we have 3 proposals for class_proposal1 and 2 proposals for class_label
        # Lets calculate the IoU for each proposal in class_proposal1 with each proposal in class_label
        # But the matching should be one to one right?
        class_iou = 0
        for proposal_item in class_proposal1:
            start1, end1 = proposal_item[1], proposal_item[2]
            best_iou = 0 # lets go with greedy approach (not one to one!)
            if(len(class_label) == 0):
                correspondences.append([proposal_item, [], 0])
                continue
            for label_item in class_label:
                start2, end2 = label_item[1], label_item[2]
                intersection = max(0, min(end1, end2) - max(start1, start2))
                union = (end1 - start1) + (end2 - start2) - intersection
                iou += intersection / union
                if(iou > best_iou):
                    best_iou = iou
            class_iou += best_iou
            correspondences.append([class_proposal1, class_label, best_iou])
        current_iou += class_iou / len(class_proposal1) # Summed best class to class IoU
    final_average_iou = current_iou / len(match_indexes) # Averaged IoU over all classes
    return final_average_iou, correspondences

def calculate_mAp_from_correspondences(correspondences, num_classes, thresholds):
    # We need to calculate the mAP for each class
    # For each class, we need to calculate the precision and recall
    # For each threshold, we need to calculate the precision and recall
    average_mAp = 0
    mAp_list = []
    for threshold in thresholds:
        class_matches = [[] for _ in range(len(num_classes))]
        class_precision = np.zeros(num_classes)
        class_recall = np.zeros(num_classes)
        for class_index in range(num_classes):
            for corresp in correspondences:
                if corresp[0][0] == class_index: # Check Class
                    if corresp[2] > threshold: # Check IoU value
                        class_matches[class_index].append(1)
                    else:
                        class_matches[class_index].append(0)
        
        for class_index in range(num_classes):
            class_precision[class_index] = np.sum(class_matches[class_index]) / len(class_matches[class_index])
            class_recall[class_index] = np.sum(class_matches[class_index]) / len(correspondences)
        mAp_list.append([np.mean(class_precision), threshold])
        average_mAp += np.mean(class_precision)
        # Now calculate the mAP for each class
    average_mAp = average_mAp / len(thresholds)
    return mAp_list, average_mAp