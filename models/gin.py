import torch
from torch_geometric.nn import GINConv, global_mean_pool, BatchNorm
import torch.nn.functional as F
import torch.nn as nn

# Improved GIN Model without LayerNorm
class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate):
        super(GIN, self).__init__()

        # First GIN layer with residual connection
        self.conv1 = GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dims[0]),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dims[0], hidden_dims[0])
            )
        )
        self.bn1 = BatchNorm(hidden_dims[0])
        self.dropout1 = nn.Dropout(dropout_rate)

        # Second GIN layer with residual connection
        self.conv2 = GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(hidden_dims[0], hidden_dims[1]),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dims[1], hidden_dims[1])
            )
        )
        self.bn2 = BatchNorm(hidden_dims[1])
        self.dropout2 = nn.Dropout(dropout_rate)

        # Third GIN layer with residual connection
        self.conv3 = GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(hidden_dims[1], hidden_dims[2]),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dims[2], hidden_dims[2])
            )
        )
        self.bn3 = BatchNorm(hidden_dims[2])
        self.dropout3 = nn.Dropout(dropout_rate)

        # Fourth GIN layer with residual connection
        self.conv4 = GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(hidden_dims[2], hidden_dims[3]),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dims[3], hidden_dims[3])
            )
        )
        self.bn4 = BatchNorm(hidden_dims[3])
        self.dropout4 = nn.Dropout(dropout_rate)

        # Fully connected layers with Dropout and residual connections
        self.fc1 = torch.nn.Linear(hidden_dims[3], hidden_dims[4])
        self.dropout_fc = nn.Dropout(dropout_rate)
        self.fc2 = torch.nn.Linear(hidden_dims[4], output_dim)

        # Linear layers for residual connections
        self.residual_fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.residual_fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.residual_fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.residual_fc4 = nn.Linear(hidden_dims[2], hidden_dims[3])

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Residual block 1
        identity = self.residual_fc1(x)
        x = F.gelu(self.bn1(self.conv1(x, edge_index)) + identity)
        x = self.dropout1(x)

        # Residual block 2
        identity = self.residual_fc2(x)
        x = F.gelu(self.bn2(self.conv2(x, edge_index)) + identity)
        x = self.dropout2(x)

        # Residual block 3
        identity = self.residual_fc3(x)
        x = F.gelu(self.bn3(self.conv3(x, edge_index)) + identity)
        x = self.dropout3(x)

        # Residual block 4
        identity = self.residual_fc4(x)
        x = F.gelu(self.bn4(self.conv4(x, edge_index)) + identity)
        x = self.dropout4(x)

        # Global mean pooling
        x = global_mean_pool(x, batch)

        # Fully connected layers
        x = F.gelu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)

        return x
