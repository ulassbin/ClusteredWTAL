import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the model directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.join(project_root, 'model'))
from model import *
import torch

def test_cluster_fusion():
    # Configuration and dummy cluster centers
    class Config:
        full_conv = True
        temporal_length = 100
        feature_dim = 5
    
    cfg = Config()
    num_cluster_centers = 3
    temporal_length = cfg.temporal_length
    feature_dim = cfg.feature_dim
    batch_size = 10

    # Dummy cluster centers (num_cluster_centers, temporal_length, feature_dim)
    cluster_centers = torch.randn(num_cluster_centers, temporal_length, feature_dim)

    # Dummy input data (batch_size, temporal_length, feature_dim)
    x = torch.randn(batch_size, temporal_length, feature_dim)

    # Dummy cluster distances (batch_size, temporal_length, num_cluster_centers)
    cluster_distances = torch.randn(batch_size, temporal_length, num_cluster_centers)

    # Initialize the model
    model = ClusterFusion(cfg, cluster_centers)

    # Test the easyFuse method
    fused_output = model.easyFuse(x, cluster_distances)
    print("Fused Output Shape:", fused_output.shape)
    print("Fused Output Data:", fused_output)

# Run the test
test_cluster_fusion()