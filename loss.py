import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ActionLoss(nn.Module):
  def __init__(self):
    super(ActionLoss, self).__init__()
    self.bce_criterion = nn.BCELoss()

  def forward(self, video_scores, label):
    #print('Video scores shape {}, label shape {}'.format(video_scores.shape, label.shape))
    label = label / torch.sum(label, dim=1, keepdim=True)
    loss = self.bce_criterion(video_scores, label)
    return loss

class ContrastiveClusterLoss(nn.Module):
  def __init__(self, margin=1.0):
    super(ContrastiveClusterLoss, self).__init__()
    self.margin = margin  # Margin for dissimilar clusters
  
  def forward(self, embeddings, cluster_centroids, labels):
    batch_size = embeddings.size(0)
    num_clusters = cluster_centroids.size(0)
    
    distances_to_clusters = torch.cdist(embeddings, cluster_centroids)  # (batch_size, num_clusters)
    
    # Compute contrastive loss
    loss = 0.0
    for i in range(batch_size):
      true_cluster = labels[i]
      positive_distance = distances_to_clusters[i, true_cluster]
      loss += positive_distance ** 2  
      for j in range(num_clusters):
        if j != true_cluster:
          negative_distance = distances_to_clusters[i, j]
          loss += F.relu(self.margin - negative_distance) ** 2      
    return loss / batch_size

class AutoLabelClusterCrossEntropyLoss(nn.Module):
  def __init__(self):
    super(AutoLabelClusterCrossEntropyLoss, self).__init__()
    self.ce_criterion = nn.CrossEntropyLoss()

  def forward(self, embeddings, cluster_centroids):
    distances_to_clusters = torch.cdist(embeddings, cluster_centroids)  # (batch_size, num_clusters)
    labels = torch.argmin(distances_to_clusters, dim=1)  # Labels are the indices of the closest clusters
    logits = -distances_to_clusters  # (batch_size, num_clusters)
    loss = self.ce_criterion(logits, labels)
    return loss, labels

class SniCoLoss(nn.Module):# Maximize distance up to a margin
  def __init__(self):
    super(SniCoLoss, self).__init__()
    self.ce_criterion = nn.CrossEntropyLoss()

  def NCE(self, q, k, neg, T=0.07):
    q = F.normalize(q, dim=1)
    k = F.normalize(k, dim=1)
    neg = neg.permute(0,2,1) # batch,time,feature -> batch,feature,time
    neg = F.normalize(neg, dim=1) # normalize from feature dimension
    l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1) 
    l_neg = torch.einsum('nc,nck->nk', [q, neg])
    logits = torch.cat([l_pos, l_neg], dim=1) # n,k+1 -> batch, (pos + negative examples) shape
    logits /= T
    labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
    loss = self.ce_criterion(logits, labels)
    return loss

  def forward(self, contrast_pairs):
    HA_refinement = self.NCE(
      torch.mean(contrast_pairs['HA'], 1), # Hard Action
      torch.mean(contrast_pairs['EA'], 1), # Easy Action
      contrast_pairs['EB'] # Easy Background
    )
    HB_refinement = self.NCE(
      torch.mean(contrast_pairs['HB'], 1), # Hard Background
      torch.mean(contrast_pairs['EB'], 1), # Easy Background
      contrast_pairs['EA'] # Easy Action
    )

    loss = HA_refinement + HB_refinement
    return loss
