import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class NearestNeighborContrastiveI3D(nn.Module):
    def __init__(self, feature_dim=2048, projection_dim=128):
        super(NearestNeighborContrastiveI3D, self).__init__()

        # Intra-video projection head
        self.intra_projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, projection_dim)
        )

        # Inter-video projection head
        self.inter_projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, projection_dim)
        )

    def forward(self, features):

        # Pass features through projection heads
        intra_embeddings = self.intra_projector(features)
        inter_embeddings = self.inter_projector(features)

        return intra_embeddings, inter_embeddings

class Queue:
    # Future work:
    # You can store videos in a queue and alongside their snippet embeddings 
    def __init__(self, queue_size=65536, embedding_dim=128):
        self.queue_size = queue_size
        self.queue = torch.zeros((queue_size, embedding_dim))  # Initialize with zeros
        # form a labels queue that is the same size as the embeddings queue but hosts the labels of type (vid_name, class_name) and type(str,str)
        self.label_queue = torch.zeros((queue_size, 2), dtype=int)
        self.vid_queue = []
        self.ptr = 0  # Pointer to track enqueueing position

    def append(self, embeddings, labels=None):
      batch_size, t, feature_dim = embeddings.shape # Assuming fixed t
      num_snips = batch_size*t
      if labels is None:
        labels = labels.reshape(-1,2)
        self.label_queue[self.ptr:self.ptr + num_snips] = copy.deepcopy(labels)

      self.queue[self.ptr:self.ptr + num_snips] = copy.deepcopy(embeddings.reshape(-1,feature_dim))
      for i in range(batch_size):
        indexes = torch.arange(self.ptr + i*t, self.ptr + i*t + t)
        self.vid_queue.append(torch.tensor(indexes, dtype=int))
      self.ptr = (self.ptr + num_snips) % self.queue_size
      print(f'Added {batch_size} embeddings to the queue size: {self.ptr}')
    
    def shift(self, amount):
        self.queue = torch.roll(self.queue, -amount, 0)
        self.label_queue = torch.roll(self.label_queue, -amount, 0)
        temp = copy.deepcopy(self.vid_queue)
        del_indexes = []

        # Correct video queue
        for i in range(len(self.vid_queue)):
          new_indexes = temp[i] - amount
          # now count valid
          valid = new_indexes[new_indexes >= 0]
          if(len(valid)==0):
            del_indexes.append(i)
          else:
            self.vid_queue[i] = valid
        for i in del_indexes:
          del self.vid_queue[i]
        self.ptr -= amount

    def enqueue(self, embeddings, labels=None):
        batch_size, temporal, feature_dim = embeddings.shape
        labels = labels.reshape(batch_size, 1, 2)
        # copy labels such that second dimension is the same as the embeddings
        labels = labels.repeat(1, temporal, 1)
        num_items = batch_size * temporal
        if self.ptr + num_items > self.queue_size:
            overflow = (self.ptr + num_items) - self.queue_size
            self.shift(overflow) # shift the queue
            self.append(embeddings, labels)
        else:
          self.append(embeddings, labels)
        print(f'Added {num_items} embeddings to the queue size: {self.ptr}')

    def dequeue(self):
        pass  # Dequeue logic automatically handled by overwriting

    def find_nearest_neighbors(self, query_embeddings):
      # Normalize embeddings to unit vectors
      query_norm = F.normalize(query_embeddings, dim=1)
      queue_norm = F.normalize(self.queue, dim=1)
      # Compute cosine similarity
      similarities = torch.matmul(query_norm, queue_norm.T)  # Shape: (batch_size, queue_size)
      # Find nearest neighbors (highest similarity)
      nn_indices = similarities.argmax(dim=1)
      return nn_indices, self.queue[nn_indices], self.label_queue[nn_indices]
    
    def sample_queue(self, amount):
        indices = torch.randperm(self.queue_size)[:amount]
        return indices, self.queue[indices], self.label_queue[indices]

    def find_nearest_neighbours_subset(self, query_embeddings, subset_size):
        small_queue_indices, small_queue, small_labels = self.sample_queue(subset_size)
        query_norm = F.normalize(query_embeddings, dim=1)
        small_queue_norm = F.normalize(small_queue, dim=1)
        similarities = torch.matmul(query_norm, small_queue_norm.T)
        nn_indices = similarities.argmax(dim=1)
        return nn_indices, small_queue[nn_indices], small_labels[nn_indices]
    
    def get_queue_without_indices(self, indices):
        mask = torch.ones(self.queue_size, dtype=bool)
        mask[indices] = False
        print(f'Got mask {mask}')
        return self.queue[mask], self.label_queue[mask]

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, query, positive, negatives):
        B, D = query.shape
        _, N, _ = negatives.shape  # B, N, D
        
        # Normalize embeddings
        query = F.normalize(query, dim=-1)      # (B, D)
        positive = F.normalize(positive, dim=-1) # (B, D)
        negatives = F.normalize(negatives, dim=-1) # (B, N, D)

        # Compute similarity scores
        pos_sim = torch.exp(torch.sum(query * positive, dim=-1) / self.temperature)  # (B,)
        neg_sim = torch.exp(torch.sum(query.unsqueeze(1) * negatives, dim=-1) / self.temperature)  # (B, N)

        # Compute InfoNCE loss
        print(f'pos_sim {pos_sim[0:5]} \nneg_sim {neg_sim.sum(dim=-1)[0:5]}')
        denominator = pos_sim + neg_sim.sum(dim=-1)  # (B,)
        loss = -torch.log(pos_sim / denominator).mean()
        return loss

def load_features(path):
    # Load features from path
    return torch.load(path)

# Example usage
if __name__ == "__main__":
    
    #features = load_features("features.pth")
    # Dummy feature input
    batch_size = 20
    temporal = 50
    feature_dim = 128
    myQueue = Queue(queue_size=50000, embedding_dim=feature_dim)

    features = torch.randn(batch_size, 100, feature_dim)
    for i in range(0,batch_size):
      labels = torch.tensor([[i,i+temporal]]*100)
      myQueue.enqueue(torch.randn(100,temporal,feature_dim), labels)

    query_snippets = features[0,30:30+batch_size,:]
    nn_indices, nn_embeddings, nn_labels = myQueue.find_nearest_neighbors(query_snippets)
    print('Got nearest neighbours {}'.format(nn_indices))
    positives = nn_embeddings
    print('Formed positives Q size {}'.format(myQueue.queue_size))
    negatives = torch.zeros(batch_size,myQueue.queue_size-1,feature_dim) # also have to remove itself
    print('Formed negatives')
    for i in range(batch_size):
      item, _ = myQueue.get_queue_without_indices(nn_indices[i]) # Copying like this might be not efficient for memory
      print(f'Got item {item.shape}')
      negatives[i] = item
    print(f'query {query_snippets.shape} positives {positives.shape} negatives {negatives.shape}')
    # test InfoNceLoss
    loss = InfoNCELoss()
    print(loss(query_snippets, positives, negatives))
    exit()

    model = NearestNeighborContrastiveI3D(feature_dim=2048, projection_dim=128)

    # Dummy input: Batch of video clips (batch_size=8, channels=3, frames=16, height=112, width=112)

    # Forward pass
    intra_embeddings, inter_embeddings = model(features)
    print(f"Intra-video embeddings shape: {intra_embeddings.shape}")
    print(f"Inter-video embeddings shape: {inter_embeddings.shape}")
