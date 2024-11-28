import numpy as np
import os
import random
from tslearn.metrics import cdist_dtw
import torch
import numpy as np
import time
import unittest
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
  return

def normalize(video):
  return (video - np.mean(video)) / np.std(video)

# Function to calculate the FFT-based distance between two videos using 2D FFT
def fft_distance_2d(video1, video2, full_conv = True):
  video1 = torch.tensor(video1.reshape(-1, feature_dim))
  video2 = torch.tensor(video2.reshape(-1, feature_dim))
  video1 = F.normalize(video1, dim=1)
  video2 = F.normalize(video2, dim=1)
  n1, d1 = video1.shape[0], video1.shape[1]
  n2, d2 = video2.shape[0], video2.shape[1]
  assert d1 == d2, "Feature dimensions must match."
  video1_fft = torch.fft.fft2(video1, dim=0)
  video2_fft = torch.fft.fft2(video2, dim=0)
  fft_mult = torch.fft.ifft2(video1_fft * torch.conj(video2_fft), dim=0)
  convolution_result = torch.real(fft_mult).sum(dim=1)
  if(not full_conv):
    peak_values = torch.amax(convolution_result)
    distances = 1 / (peak_values + 1e-10)
  else:
    distances = 1 / (convolution_result + 1e-10)
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

def circular_cosine_similarity(video_batch1, video_batch2):
  assert video_batch1.shape == video_batch2.shape, "Both video batches must have the same shape"
  batch_size, time_dim, feature_dim = video_batch1.shape
  video_batch1_norm = F.normalize(video_batch1, dim=2)
  video_batch2_norm = F.normalize(video_batch2, dim=2)
  result = torch.zeros((batch_size, time_dim))
  for shift in range(time_dim):
    shifted_video_batch2 = torch.roll(video_batch2_norm, shifts=shift, dims=1)
    cosine_sim = torch.sum(video_batch1_norm * shifted_video_batch2, dim=2)
    result[:, shift] = cosine_sim.sum(dim=1)
  return result

def plot_comparison(convolution_result, shifted_sum_result):
  num_plots = convolution_result.shape[0]
  num_cols = math.ceil(math.sqrt(num_plots))
  num_rows = math.ceil(num_plots / num_cols)
  fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))
  axes = axes.flatten()
  for i in range(num_plots):
    ax = axes[i]
    ax.plot(convolution_result[i].cpu().numpy(), label=f'FFT Result {i+1}', color='b')
    ax.plot(shifted_sum_result[i].cpu().numpy(), label=f'ShiftedSum Result {i+1}', color='r', linestyle='--')
    ax.set_xlabel('X-axis (Time)')
    ax.set_ylabel('Y-axis (Values)')
    ax.set_title(f'Comparison for Index {i+1}')
    ax.legend()
  for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
  plt.tight_layout()
  plt.show()


def normalize(video_batch):
  return (video_batch - video_batch.mean(dim=(1,2), keepdim=True)) / video_batch.std(dim=(1,2), keepdim=True)

def fft_distance_2d_batch(video_batch1, video_batch2, full_conv=True):
  video_batch1 = F.normalize(video_batch1, dim=2)
  video_batch2 = F.normalize(video_batch2, dim=2)
  n1, d1 = video_batch1.shape[1], video_batch1.shape[2]
  n2, d2 = video_batch2.shape[1], video_batch2.shape[2]
  assert d1 == d2, "Feature dimensions must match."
  video_batch1_fft = torch.fft.fft2(video_batch1, dim=1)
  video_batch2_fft = torch.fft.fft2(video_batch2, dim=1)
  fft_mult = torch.fft.ifft2(video_batch1_fft * torch.conj(video_batch2_fft), dim=1)
  convolution_result = torch.real(fft_mult).sum(dim=2)
  if not full_conv:
    peak_values = torch.amax(convolution_result)
    distances = 1 / (peak_values + 1e-10)
  else:
    distances = 1 / (convolution_result + 1e-10)
  return distances

def cdist_fft_2d_batched(videos, batch_size=32, full_conv=False):
  num_videos = len(videos)
  distance_matrix = torch.zeros((num_videos, num_videos), device=videos[0].device)
  with torch.no_grad():
    for i in range(num_videos):
      for j in range(i + 1, num_videos, batch_size):
        end = min(j + batch_size, num_videos)
        batch_videos_j = videos[j:end].to('cuda')
        batch_videos_i = videos[i].unsqueeze(0).expand(end - j, -1, -1).to('cuda')
        torch.cuda.empty_cache()
        distances = fft_distance_2d_batch(batch_videos_i, batch_videos_j, full_conv)
        torch.cuda.empty_cache()
        distance_matrix[i, j:end] = distances
        distance_matrix[j:end, i] = distances
      torch.cuda.empty_cache()
      print_memory_usage('{} Iteration'.format(i))
  return distance_matrix


def cdist_fft_2d_batched_filenames(filenames, data_loader, batch_size=32, full_conv=False):
  num_videos = len(filenames)
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  distance_matrix = torch.zeros((num_videos, num_videos), device=device)
  with torch.no_grad():
    for i in range(num_videos):
      for j in range(i + 1, num_videos, batch_size):
        progress = (i + 1) * (j / batch_size) / (num_videos * num_videos / batch_size)
        print('Progress {}'.format(progress * 100.0))
        end = min(j + batch_size, num_videos)
        videos_j = data_loader.load_mini_batch(filenames[j:end])
        videos_i = data_loader.load_mini_batch([filenames[i]] * (end - j))
        batch_videos_j = torch.tensor(videos_j).to('cuda').reshape(-1, data_loader.max_len, data_loader.feature_dim)
        batch_videos_i = torch.tensor(videos_i).to('cuda').reshape(-1, data_loader.max_len, data_loader.feature_dim)
        distances = fft_distance_2d_batch(batch_videos_i, batch_videos_j, full_conv)
        distance_matrix[i, j:end] = distances
        distance_matrix[j:end, i] = distances
      torch.cuda.empty_cache()
      print_memory_usage('{} Iteration'.format(i))
  return distance_matrix

def cuda_fft_distances(filenames, data_loader,  feature_size, batch=32):
  num_videos = len(filenames)
  distance_matrix = cdist_fft_2d_batched_filenames(filenames, data_loader, batch_size=batch)
  return distance_matrix.to('cpu').numpy()

def compute_cluster_distances(x, cluster_centers):
  batch_size, temporal_length, feature_dim = x.shape
  centers, cent_length, cent_feature_dim = cluster_centers.shape
  assert temporal_length == cent_length, "Temporal lengths must match between x and cluster centers."
  assert feature_dim == cent_feature_dim, "Feature dimensions must match between x and cluster centers."
  x_expanded = x.unsqueeze(1).expand(batch_size, centers, temporal_length, feature_dim).reshape(-1, temporal_length, feature_dim)
  cluster_centers_expanded = cluster_centers.unsqueeze(0).expand(batch_size, centers, temporal_length, feature_dim).reshape(-1, temporal_length, feature_dim)
  distances_expanded = fft_distance_2d_batch(x_expanded, cluster_centers_expanded)
  distances = distances_expanded.view(batch_size, centers, temporal_length).permute(0, 2, 1).to('cuda')
  return distances


#Iou and mAP calculation below, later we group those into a class!

class OverlapWizard:
  def __init__(self, batch_size, num_classes):
    self.correspondences = []
    self.batch_size = batch_size
    self.num_classes = num_classes

  def get_potential_matches(self, proposal, proposal_label):
    batch_size = len(proposal)
    potential_matches = [[] for _ in range(batch_size)]
    for batch_index in range(batch_size):
      batch_proposal = proposal[batch_index]
      batch_label = proposal_label[batch_index]
      for classf in range(len(batch_proposal)):
        #print(' Prop {} \n label {}'.format(batch_proposal[classf], batch_label[classf]))
        if not batch_proposal[classf] or not batch_label[classf]:
          continue
        potential_matches[batch_index].append(classf)
    return potential_matches, np.sum([len(x) for x in potential_matches])

  def findBestMatch(self, proposal_item, c_label): # In future drop c_labels after matching. Make it one to one!
    start1, end1 = proposal_item[1], proposal_item[2]
    best_iou = 0
    best_label = None
    eps = 1e-5
    for label_item in c_label:
      start2, end2 = label_item[1], label_item[2]
      intersection = max(0, min(end1, end2) - max(start1, start2))
      union = (end1 - start1) + (end2 - start2) - intersection
      iou = intersection / (union + eps)
      if iou > best_iou:
        best_iou = iou
        best_label = label_item
    return best_iou, best_label

  #def getMatches(self, potential_matches, b_proposal, b_proposal_label): # batch proposal and batch proposal label
  #  classwise_iou = [[0, 0] for _ in range(self.num_classes)]
  #  for class_index in potential_matches:
  #    print('pmatches {}, cindex {}, num_class {}'.format(len(potential_matches), class_index, self.num_classes))
  #    c_proposal = b_proposal[class_index]
  #    c_label = b_proposal_label[class_index]
  #    if not c_proposal or not c_label:
  #      continue
  #    class_iou = 0 
  #    for proposal_item in c_proposal:
  #      best_iou, best_label = self.findBestMatch(proposal_item, c_label)
  #      class_iou += best_iou
  #      self.correspondences.append([proposal_item, best_label, best_iou])
  #    max_len = max(len(c_proposal), len(c_label))
  #    class_iou = class_iou / max_len if max_len > 0 else 0
  #    classwise_iou[class_index] = [class_iou, max_len]
  #  return classwise_iou

  def getMatches(self, potential_matches, b_proposal, b_proposal_label): # debug version
    classwise_iou = [[0, 0] for _ in range(self.num_classes)]
    print("Potential matches:", potential_matches)
    #print("Batch proposals:", b_proposal)
    #print("Batch proposal labels:", b_proposal_label)
    for class_index in potential_matches:
        print('Processing class_index:', class_index)
        c_proposal = b_proposal[class_index]
        c_label = b_proposal_label[class_index]
        if not c_proposal or not c_label:
            print(f"Skipping class_index {class_index}: c_proposal empty: {not c_proposal}, c_label empty: {not c_label}")
            continue
        class_iou = 0
        for proposal_item in c_proposal:
            best_iou, best_label = self.findBestMatch(proposal_item, c_label)
            print(f"Proposal item: {proposal_item}, Best IOU: {best_iou}, Best label: {best_label}")
            class_iou += best_iou
            self.correspondences.append([proposal_item, best_label, best_iou])
        max_len = max(len(c_proposal), len(c_label))
        if max_len > 0:
            class_iou = class_iou / max_len
        else:
            print(f"Warning: max_len is 0 for class_index {class_index}")
            class_iou = 0
        print(f"Class Index: {class_index}, Class IOU: {class_iou}, Max Length: {max_len}")
        classwise_iou[class_index] = [class_iou, max_len]
    return classwise_iou

  def resetCorrespondences(self):
    self.correspondences = []
    return

  def get_classwise_average(self, classwise_iou): # work on this!!
    sum = 0
    count = 0
    for class_iou, max_len in classwise_iou:
      curr_iou = class_iou
      if max_len == 0:
        curr_iou = 0
      sum += curr_iou * max_len
      count += max_len
    print('Count ', count)
    return sum / count if count > 0 else 0 # get back to here. -1 is just an indicator

  def calculate_IoU(self, proposal, proposal_label):
    # Proposals are of shape (batch_size, num_classes, num_proposals)
    # And each proposal is of shape [class_index, start, end, threshold, score]
    self.resetCorrespondences()
    assert len(proposal) == len(proposal_label), 'batch sizes for proposals dont match'
    batch_size = len(proposal_label)
    if not proposal or not proposal_label:
      print('Empty prop {} or label {}'.format(len(proposal), len(proposal_label)))
      return np.zeros(batch_size), self.correspondences

    potential_matches, max_matches = self.get_potential_matches(proposal, proposal_label)
    if(max_matches == 0):
      print('Potential matches is zero')
      return np.zeros(batch_size), self.correspondences

    final_average_iou = np.zeros(batch_size)
    print('Batch size here is ', batch_size)
    for batch_index in range(batch_size):
      classwise_iou = self.getMatches(potential_matches[batch_index], proposal[batch_index], proposal_label[batch_index])
      final_average_iou[batch_index] = self.get_classwise_average(classwise_iou)
      print('Batch {} Class wise IoU \n {}'.format(batch_index, classwise_iou))
    return final_average_iou, self.correspondences

  def calculate_mAp_from_correspondences(self, thresholds):
    average_mAp = 0
    mAp_list = []
    for threshold in thresholds:
      class_matches = [[] for _ in range(self.num_classes)]
      class_precision = np.zeros(self.num_classes)
      class_recall = np.zeros(self.num_classes) # implement moving on.
      for corresp in self.correspondences:
        if corresp[2] > threshold:
          class_matches[corresp[0][0]].append(1)
        else:
          class_matches[corresp[0][0]].append(0)
      class_precision = 0
      for class_index in range(self.num_classes):
        if not class_matches[class_index]:
          continue
        tp = np.sum(class_matches[class_index])
        fp = len(class_matches[class_index]) - tp
        class_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
      mAp_list.append([np.mean(class_precision), threshold])
      average_mAp += np.mean(class_precision)
    average_mAp = average_mAp / len(thresholds)
    return mAp_list, average_mAp

# Define a test function to test the above class, you can use unit test
class testOverlapWizard(unittest.TestCase):

  def test_findBestMatch(self):
    # proposals are of shape (batch_size, num_classes, num_proposals)
    # and each item is of shape [class_index, start, end, threshold, score]
    num_classes = 5
    batch_size = 3
    thresholds = [0.25, 0.5, 0.75,]
    proposal = [[[[0, 0, 10, 0.5, 0.8], [0, 10, 20, 0.5, 0.8], [0, 20, 30, 0.5, 0.8]],
                 [[1, 0, 10, 0.5, 0.8], [1, 10, 20, 0.5, 0.8], [1, 20, 30, 0.5, 0.8]],
                 [[2, 0, 10, 0.5, 0.8], [2, 10, 20, 0.5, 0.8], [2, 20, 30, 0.5, 0.8]]],
                 [[] for _ in range(num_classes)],
                 [[], [], []]]
    label    = [[[[0, 0, 15, 0.5, 1.8], [0, 10, 25, 0.75, 0.8], [0, 20, 30, 0.75, 0.8]],
                 [[1, 0, 13, 0.5, 1.8], [1, 10, 25, 0.75, 0.8], [1, 20, 30, 0.75, 0.8]],
                 [[2, 0, 15, 0.5, 1.8], [2, 15, 20, 0.75, 0.8], [2, 20, 30, 0.75, 0.8]]],
                 [[] for _ in range(num_classes)],
                 [[], [], []]]
    print('Testing Overlap Wizard')
    ow = OverlapWizard(batch_size, num_classes)
    average_iou, correspondences = ow.calculate_IoU(proposal, label)
    print('Average IoU: ', average_iou)
    print('Correspondences: ', correspondences)
    mAp_list, average_mAp = ow.calculate_mAp_from_correspondences([0.25, 0.5, 0.75])
    print('mAp List: ', mAp_list)
    print('Average mAp: ', average_mAp)
    return
  
if __name__ == '__main__':
  unittest.main()
  print('All tests Done')
