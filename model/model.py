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


class ClusterModule(nn.Module):
    def __init__(self, cluster_data, cfg):
        super(ClusterModule, self).__init__()
        self.full_distance = cfg.distance
        self.clusters = cluster_data
        self.full_conv = cfg.full_conv

    def forward(self, x):
        # Full conv distance (#xBatch, #clusterCenters, #temporalLength)
        # Peak conv distance (#xBatch, #clusterCenters, 1)
        distances = []
        helper.full_conv = self.full_conv
        for center in cluster_data:
            dist = helper.fft_distance_2d(x) # Change this to batched.
            distances.append(dist)
        return distances



class ClusterFusion(nn.Module):
    # After computing the full convolutions, we want to affect 
    def __init__(self, cfg, cluster_centers):
        super(ClusterFusion, self).__init__()
        self.full_conv = cfg.full_conv
        self.temporal_length = cfg.temporal_length
        self.feature_dim = cfg.feature_dim
        self.scale = nn.Parameter(torch.ones(1)) # Learnable scale parameters
        self.cluster_centers = cluster_centers # This is similar vectors with x.
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



    def forward(x, cluster_distances):
        # Cluster distances shape (#xBatch, #temporalLength, #clusterCenters)
        # X(batched videos) shape (#xBatch, #temporalLength, #featureDimension)
        # Cluster distances are convolutions - time frame shift based similarities of X vectors of videos from
        # cluster centers. If cluster distances are of an x item is small for time =t and center =1. It is the most 
        # corresponding video when x is shifted t time steps and overlapped with clustercenter =1 

        # Use attention based similarity if possible, how to incorporate this distances to fuse to X vectors,
        # While keeping dimension
        return 0



class CrashingVids(nn.Module):
    def __init__(self, cfg):
        super(CrashingVids, self).__init__()
        self.len_feature = cfg.FEATS_DIM
        self.num_classes = cfg.NUM_CLASSES

        self.actionness_module = Simple_Actionness_Module(cfg.FEATS_DIM, cfg.NUM_CLASSES)

        self.softmax = nn.Softmax(dim=1)
        self.softmax_2 = nn.Softmax(dim=2)

    def loadClusters():
        pass
    def getClusterDistance(x):
        dist = 0
        return dist
    def select_topk_embeddings(self, scores, embeddings, k):
        _, idx_DESC = scores.sort(descending=True, dim=1)
        idx_topk = idx_DESC[:, :k]
        idx_topk = idx_topk.unsqueeze(2).expand([-1, -1, embeddings.shape[2]])
        selected_embeddings = torch.gather(embeddings, 1, idx_topk)
        return selected_embeddings

    def easy_snippets_mining(self, actionness, embeddings, k_easy):
        select_idx = torch.ones_like(actionness).cuda()
        select_idx = self.dropout(select_idx)

        actionness_drop = actionness * select_idx

        actionness_rev = torch.max(actionness, dim=1, keepdim=True)[0] - actionness
        actionness_rev_drop = actionness_rev * select_idx

        easy_act = self.select_topk_embeddings(actionness_drop, embeddings, k_easy)
        easy_bkg = self.select_topk_embeddings(actionness_rev_drop, embeddings, k_easy)

        return easy_act, easy_bkg

    def hard_snippets_mining(self, actionness, embeddings, k_hard):
        aness_np = actionness.cpu().detach().numpy()
        aness_median = np.median(aness_np, 1, keepdims=True)
        aness_bin = np.where(aness_np > aness_median, 1.0, 0.0)

        erosion_M = ndimage.binary_erosion(aness_bin, structure=np.ones((1,self.M))).astype(aness_np.dtype)
        erosion_m = ndimage.binary_erosion(aness_bin, structure=np.ones((1,self.m))).astype(aness_np.dtype)
        idx_region_inner = actionness.new_tensor(erosion_m - erosion_M)
        aness_region_inner = actionness * idx_region_inner
        hard_act = self.select_topk_embeddings(aness_region_inner, embeddings, k_hard)

        dilation_m = ndimage.binary_dilation(aness_bin, structure=np.ones((1,self.m))).astype(aness_np.dtype)
        dilation_M = ndimage.binary_dilation(aness_bin, structure=np.ones((1,self.M))).astype(aness_np.dtype)
        idx_region_outer = actionness.new_tensor(dilation_M - dilation_m)
        aness_region_outer = actionness * idx_region_outer
        hard_bkg = self.select_topk_embeddings(aness_region_outer, embeddings, k_hard)

        return hard_act, hard_bkg

    def get_video_cls_scores(self, cas, k_easy):
        sorted_scores, _= cas.sort(descending=True, dim=1)
        topk_scores = sorted_scores[:, :k_easy, :]
        video_scores = self.softmax(topk_scores.mean(1))
        return video_scores

    def forward(self, x):
        num_segments = x.shape[1]
        k_easy = num_segments // self.r_easy
        k_hard = num_segments // self.r_hard
        embeddings, cas, actionness = self.actionness_module(x)

        easy_act, easy_bkg = self.easy_snippets_mining(actionness, embeddings, k_easy)
        hard_act, hard_bkg = self.hard_snippets_mining(actionness, embeddings, k_hard)
        
        video_scores = self.get_video_cls_scores(cas, k_easy)

        contrast_pairs = {
            'EA': easy_act,
            'EB': easy_bkg,
            'HA': hard_act,
            'HB': hard_bkg
        }

        return video_scores, contrast_pairs, actionness, cas

























def RSKP_PredictionModule(self, x):
    # normalization
    norms_x = calculate_l1_norm(x)
    norms_ac = calculate_l1_norm(self.ac_center)
    norms_fg = calculate_l1_norm(self.fg_center)

    # generate class scores
    frm_scrs = torch.einsum('ntd,cd->ntc', [norms_x, norms_ac]) * self.scale_factor
    frm_fb_scrs = torch.einsum('ntd,kd->ntk', [norms_x, norms_fg]).squeeze(-1) * self.scale_factor

    # generate attention
    class_agno_att = self.sigmoid(frm_fb_scrs)
    class_wise_att = self.sigmoid(frm_scrs)
    class_agno_norm_att = class_agno_att / (torch.sum(class_agno_att, dim=1, keepdim=True) + 1e-5)
    class_wise_norm_att = class_wise_att / (torch.sum(class_wise_att, dim=1, keepdim=True) + 1e-5)

    ca_vid_feat = torch.einsum('ntd,nt->nd', [x, class_agno_norm_att])
    cw_vid_feat = torch.einsum('ntd,ntc->ncd', [x, class_wise_norm_att])

    # normalization
    norms_ca_vid_feat = calculate_l1_norm(ca_vid_feat)
    norms_cw_vid_feat = calculate_l1_norm(cw_vid_feat)

    # classification
    frm_scr = torch.einsum('ntd,cd->ntc', [norms_x, norms_ac]) * self.scale_factor
    ca_vid_scr = torch.einsum('nd,cd->nc', [norms_ca_vid_feat, norms_ac]) * self.scale_factor
    cw_vid_scr = torch.einsum('ncd,cd->nc', [norms_cw_vid_feat, norms_ac]) * self.scale_factor

    # prediction
    ca_vid_pred = F.softmax(ca_vid_scr, -1)
    cw_vid_pred = F.softmax(cw_vid_scr, -1)

    return ca_vid_pred, cw_vid_pred, class_agno_att, frm_scr