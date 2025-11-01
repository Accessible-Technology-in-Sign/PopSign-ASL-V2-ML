import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class SimpleGNN(nn.Module):
    def __init__(self, max_frames, input_features, num_coords, num_signs):
        """
        Args:
            max_frames: Hidden dimension size for GCN layers.
            input_features: Number of features per coordinate (e.g., xyz -> 3).
            num_coords: Number of coordinates (nodes) in the graph.
            num_signs: Number of output classes for classification.
        """
        super(SimpleGNN, self).__init__()
        
        self.num_nodes = num_coords  # Number of nodes
        self.node_features = input_features  # Features per node

        # Graph Convolutional layers
        self.conv1 = GCNConv(input_features, max_frames)
        self.conv2 = GCNConv(max_frames, max_frames)
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers for classification
        self.fc1 = nn.Linear(max_frames, max_frames)
        self.fc2 = nn.Linear(max_frames, num_signs)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, num_features, num_coords]
            where:
            - batch_size: Number of samples in the batch.
            - seq_len: Number of frames (temporal dimension).
            - num_features: Number of features (e.g., xyz coordinates).
            - num_coords: Number of coordinates (nodes in the graph).

        Returns:
            logits: Output tensor of shape [batch_size, num_signs].
        """
        batch_size, seq_len, num_features, num_coords = x.size()

        # Reshape to treat each coordinate as a node
        x = x.permute(0, 3, 1, 2).contiguous()  # [batch_size, num_coords, seq_len, num_features]
        x = x.view(-1, seq_len, num_features)  # Flatten batch and nodes: [batch_size * num_coords, seq_len, num_features]

        # Create a simple edge index for a fully connected graph within each seq_len
        edge_index = torch.combinations(torch.arange(seq_len), r=2).T.to(x.device)  # [2, num_edges]

        # Repeat edge_index for all nodes in the batch
        edge_index = edge_index.repeat(batch_size * num_coords, 1)

        # Construct batch assignments
        graph_batch = torch.arange(batch_size * num_coords).repeat_interleave(seq_len).to(x.device)

        # Graph Convolutional Layers
        x = x.view(-1, num_features)  # [num_coords * batch_size * seq_len, num_features]
        print(x.size())
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Global pooling to summarize temporal information
        x = global_mean_pool(x, graph_batch)  # [batch_size * num_coords, max_frames]

        # Reshape back to batch and coordinates
        x = x.view(batch_size, num_coords, -1)  # [batch_size, num_coords, max_frames]

        # Pool across coordinates to form graph embeddings for the entire graph
        x = x.mean(dim=1)  # [batch_size, max_frames]

        # Dropout regularization
        x = self.dropout(x)

        # Fully connected layers for classification
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)  # [batch_size, num_signs]

        return x

