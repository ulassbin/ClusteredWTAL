import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors

# Example feature data
data_dir = '/home/ulas/Documents/Datasets/CoLA/data/THUMOS14/features/test/rgb/'
file = 'video_test_0000437.npy'
features = np.load(data_dir+file) 

# Reduce features
features = features[0:50,...]
# Initialize the KNN model
print('Data size ', features.shape)
knn = NearestNeighbors(n_neighbors=50, algorithm='ball_tree').fit(features)

# Find the 5 nearest neighbors for each point in the dataset
distances, indices = knn.kneighbors(features)

# Create a matrix to mark the nearest neighbors
nn_matrix = np.zeros((features.shape[0], features.shape[0]))

# Fill the matrix with marks (1) where there are nearest neighbor relationships
for i in range(len(indices)):
    for j in range(len(indices[i])):
        neighbor = indices[i][j]
        nn_matrix[i, neighbor] = distances[i][j]
nn_matrix = nn_matrix / np.sum(nn_matrix)
# Plot the matrix as a heatmap
plt.figure(figsize=(10, 8))
plt.imshow(nn_matrix, cmap='viridis', aspect='auto')
plt.colorbar(label='Nearest Neighbor')
plt.title('Nearest Neighbors Heatmap')
plt.xlabel('Frame index')
plt.ylabel('Frame index')

# Overlay 'x' marks for the nearest neighbors
for i in range(len(indices)):
    for neighbor in indices[i]:
        plt.text(neighbor, i, 'x', color='red', ha='center', va='center', fontsize=8)

plt.show()
