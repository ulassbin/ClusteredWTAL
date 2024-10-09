from sklearn.metrics import silhouette_samples
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import time
import random

def getClusters(data, K):
    # Perform K means
    kmeans = KMeans(n_clusters=K, random_state=0)
    labels = kmeans.fit_predict(data)

def average_silhouette_scores_per_cluster(data, labels):
    silhouette_vals = silhouette_samples(data, labels)
    unique_labels = np.unique(labels)
    silhouette_per_cluster = {}
    
    for label in unique_labels:
        cluster_silhouette_vals = silhouette_vals[labels == label]
        silhouette_per_cluster[label] = np.mean(cluster_silhouette_vals)
    
    return silhouette_per_cluster

def plot_silhouette_scores(silhouette_per_cluster, i):
    plt.figure(figsize=(8, 5))
    clusters = list(silhouette_per_cluster.keys())
    scores = list(silhouette_per_cluster.values())
    plt.bar(clusters, scores)
    plt.xlabel('Cluster')
    plt.ylabel('Average Silhouette Score')
    plt.title('Average Silhouette Score {}'.format(i))
    #plt.show()


def elbow_and_silhoutte(data, max_clusters=10, visualize=False):
    start_time = time.time()
    inertia = []
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=0)
        labels = kmeans.fit_predict(data)
        inertia.append(kmeans.inertia_)
        sil = average_silhouette_scores_per_cluster(data, labels)
        if(visualize):
            plot_silhouette_scores(sil, k)
    end_time = time.time()
    duration = end_time - start_time
    print(f"Analysis Calculation took {duration:.2f} seconds to execute.")
    if(visualize):
        plt.figure(figsize=(8, 5))
        plt.plot(range(2, max_clusters + 1), inertia, 'bx-')
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia')
        plt.title('Elbow Method for Optimal k')
        plt.show()

