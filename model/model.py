import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import helper_functions as helper
from logger import logging, CustomLogger

print('Loaded model!')
modelLogger = CustomLogger(name='main')
# (a) Feature Embedding and (b) Actionness Modeling
class Simple_Actionness_Module(nn.Module):
    def __init__(self, len_feature, num_classes):
        super(Simple_Actionness_Module, self).__init__()
        self.len_feature = len_feature
        self.f_embed = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature, out_channels=2048, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU()
        )

        self.f_cls = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=num_classes, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(p=0.7)

    def forward(self, x):
        out = x.permute(0, 2, 1)
        out = self.f_embed(out)
        embeddings = out.permute(0, 2, 1)
        out = self.dropout(out)
        out = self.f_cls(out)
        cas = out.permute(0, 2, 1)
        actionness = cas.sum(dim=2)
        return embeddings, cas, actionness

class ClusterFusion(nn.Module):
    # After computing the full convolutions, we want to affect 
    def __init__(self, cfg, cluster_centers):
        super(ClusterFusion, self).__init__()
        self.temporal_length = cfg.TEMPORAL_LENGTH
        self.feature_dim = cfg.FEATS_DIM
        self.num_cluster = cluster_centers.shape[0]
        self.scale = nn.Parameter(torch.ones(1)).to('cuda') # Learnable scale parameters
        self.cluster_centers = cluster_centers.view(self.num_cluster, self.temporal_length, -1) # This is similar vectors with x.
        # cluster_centers shape (#clustercenters, #temporalLength, #featureDimension)
        # Cluster distances shape (#batch, #temporalLength, #clusterCenters)
        # X(batched videos) shape (#batch, #temporalLength, #featureDimension)
        # We want to have something like x_fused = x + cluster_dist * cluster_center

    def easyFuse(self, x, cluster_distances):
        # x shape: (batch_size, temporal_length, feature_dim)
        # cluster_distances shape: (batch_size, temporal_length, num_cluster_centers)
        batch_size, temporal_length, feature_dim = x.shape
        n_centers, _, _ = self.cluster_centers.shape

        # Normalize the cluster distances based on time and cluster center
        dist_norm = F.normalize(cluster_distances, dim=(1, 2))  # Normalize across time and cluster centers

        # Expand dimensions for broadcasting
        information = (self.cluster_centers.unsqueeze(0) * 
                       dist_norm.reshape(batch_size, n_centers, temporal_length).unsqueeze(-1))
        # information shape: (batch_size, n_centers, temporal_length, feature_dim)

        # Sum over the cluster centers dimension
        information = torch.sum(information, dim=1)  # Shape: (batch_size, temporal_length, feature_dim)

        # Fused data using a scaling factor (like a Kalman gain type of parameter)
        fused_data = x + self.scale.view(1, 1, -1) * information  # Broadcasting the scale parameter
        return fused_data


    def compute_distances(self, x, cluster_centers):
        helper.full_conv = True
        batch_size, temporal_length, feature_dim = x.shape  # (batch_size, temporal_length, feature_dim)
        centers, cent_length, cent_feature_dim = cluster_centers.shape  # (centers, temporal_length, feature_dim)

        # Ensure that the temporal and feature dimensions of x and cluster centers match
        assert temporal_length == cent_length, "Temporal lengths must match between x and cluster centers."
        assert feature_dim == cent_feature_dim, "Feature dimensions must match between x and cluster centers."

        # Expand x to (batch_size * centers, temporal_length, feature_dim)
        x_expanded = x.unsqueeze(1).expand(batch_size, centers, temporal_length, feature_dim).reshape(-1, temporal_length, feature_dim)

        # Expand cluster_centers to (batch_size * centers, temporal_length, feature_dim)
        cluster_centers_expanded = cluster_centers.unsqueeze(0).expand(batch_size, centers, temporal_length, feature_dim).reshape(-1, temporal_length, feature_dim)

        # Compute distances using fft_distance_2d_batch; the result will be (batch_size * centers, temporal_length)
        distances_expanded = helper.fft_distance_2d_batch(x_expanded, cluster_centers_expanded)  # Shape: (batch_size * centers, temporal_length)

        # Reshape the distances to (batch_size, centers, temporal_length) and then transpose to (batch_size, temporal_length, centers)
        distances = distances_expanded.view(batch_size, centers, temporal_length).permute(0, 2, 1).to('cuda')  # Shape: (batch_size, temporal_length, centers)

        return distances

    def forward(self, x):
        distances = self.compute_distances(x, self.cluster_centers)
        print("X {}, distances {}".format(x.device, distances.device))
        return self.easyFuse(x, distances), distances

class TransformerClusterFusion(nn.Module):
    def __init__(self, cfg, cluster_centers):
        super(TransformerClusterFusion, self).__init__()
        self.temporal_length = cfg.temporal_length
        self.feature_dim = cfg.feature_dim
        self.num_cluster_centers = cluster_centers.shape[0]
        self.scale = nn.Parameter(torch.ones(1)).to('cuda')  # Learnable scale parameter
        self.cluster_centers = cluster_centers  # Shape: (num_cluster_centers, temporal_length, feature_dim)

        # Transformer Encoder Layer
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.feature_dim,
            nhead=8,  # Number of attention heads
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=4)

        # Linear layer to project information after transformer
        self.projection = nn.Linear(cfg.feature_dim, cfg.feature_dim)

    def transformerFuse(self, x, cluster_distances):
        """
        Args:
            x: Input video feature tensor of shape (batch_size, temporal_length, feature_dim).
            cluster_distances: Tensor representing the distances between each video feature and cluster centers,
                               of shape (batch_size, temporal_length, num_cluster_centers).
        Returns:
            Fused feature tensor of shape (batch_size, temporal_length, feature_dim).
        """
        batch_size, temporal_length, feature_dim = x.shape
        num_cluster_centers, _, _ = self.cluster_centers.shape

        # Step 1: Normalize the cluster distances
        dist_norm = F.normalize(cluster_distances, dim=(1, 2))
        # dist_norm shape: (batch_size, temporal_length, num_cluster_centers)

        # Step 2: Expand cluster_centers for batch compatibility
        cluster_centers_expanded = self.cluster_centers.unsqueeze(0)  # Shape: (1, num_cluster_centers, temporal_length, feature_dim)
        cluster_centers_expanded = cluster_centers_expanded.expand(batch_size, -1, -1, -1)
        # cluster_centers_expanded shape: (batch_size, num_cluster_centers, temporal_length, feature_dim)

        # Step 3: Expand dist_norm for broadcasting
        dist_norm_expanded = dist_norm.unsqueeze(-1)  # Shape: (batch_size, temporal_length, num_cluster_centers, 1)
        dist_norm_expanded = dist_norm_expanded.permute(0, 2, 1, 3)
        # dist_norm_expanded shape: (batch_size, num_cluster_centers, temporal_length, 1)

        # Step 4: Calculate weighted information
        information = dist_norm_expanded * cluster_centers_expanded
        # information shape: (batch_size, num_cluster_centers, temporal_length, feature_dim)

        # Step 5: Sum information over cluster centers
        aggregated_information = information.sum(dim=1)  # Shape: (batch_size, temporal_length, feature_dim)

        # Step 6: Concatenate x and aggregated information for transformer input
        fused_input = x + self.scale.view(1, 1, -1) * aggregated_information
        # fused_input shape: (batch_size, temporal_length, feature_dim)

        # Step 7: Pass through Transformer Encoder
        transformer_output = self.transformer_encoder(fused_input)  # Shape: (batch_size, temporal_length, feature_dim)

        # Step 8: Project the transformer output back to original feature dimension
        fused_data = self.projection(transformer_output)  # Shape: (batch_size, temporal_length, feature_dim)

        return fused_data

class CrashingVids(nn.Module):
    def __init__(self, cfg, temporal_length, cluster_centers):
        super(CrashingVids, self).__init__()
        cfg.TEMPORAL_LENGTH = temporal_length
        self.temporal_length = cfg.TEMPORAL_LENGTH
        self.feature_dim = cfg.FEATS_DIM
        self.scale = nn.Parameter(torch.ones(1)) # Learnable scale parameters
        self.cluster_centers = cluster_centers # This is similar vectors with x.

        self.softmax = nn.Softmax(dim=1)
        self.actionness_module = Simple_Actionness_Module(cfg.FEATS_DIM, cfg.NUM_CLASSES).to('cuda') # TransformerClusterFusion
        self.cluster_fusion_module = ClusterFusion(cfg, cluster_centers).to('cuda')
    
    def get_video_cls_scores(self, cas, filtered=None):
        # If no filtered indexes are provided, use the full temporal length
        if filtered is None:
            filtered = torch.arange(cas.size(1), device=cas.device)  # Full range of temporal length

        filtered_cas = cas[:, filtered, :]  # Shape: (batch_size, filtered_length, num_classes)
        avg_scores = filtered_cas.mean(dim=1)  # Shape: (batch_size, num_classes)
        video_scores = self.softmax(avg_scores)
        return video_scores


    def forward(self, x):
        print('X shape ', x.shape)
        x = x.reshape(-1,self.temporal_length, self.feature_dim)
        batch_size, temporal_length, feature_dim = x.shape

        x_fused, distances = self.cluster_fusion_module(x)
        print('X fused is ', type(x_fused), ' Shape is ', x_fused.shape)
        embeddings, cas, actionness = self.actionness_module(x_fused)
        base_embed, base_cas, base_actionnes = self.actionness_module(x)

        video_scores = self.get_video_cls_scores(cas)
        base_vid_scores = self.get_video_cls_scores(base_cas)
        return video_scores, actionness, cas, base_vid_scores, base_actionnes, base_cas


if __name__=="__main__":

    ModelMain = CrashingVids(cfg, cluster_centers)