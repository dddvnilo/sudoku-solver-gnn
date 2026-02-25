import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class SudokuGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_action_types, num_cells, num_numbers, num_techniques):
        super().__init__()
        
        # -------- GNN layers --------
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        # -------- MLP heads za predikcije --------
        self.action_head = nn.Linear(hidden_channels, num_action_types)
        self.cell_head = nn.Linear(hidden_channels, num_cells)
        self.number_head = nn.Linear(hidden_channels, num_numbers)
        self.technique_head = nn.Linear(hidden_channels, num_techniques)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, getattr(data, 'batch', None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)  # ako nije batch
        
        # -------- GNN forward --------
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        
        # -------- global pooling po grafu (sve celije) --------
        x_pool = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        
        # -------- predikcije --------
        y_action = self.action_head(x_pool)
        y_cell = self.cell_head(x_pool)
        y_number = self.number_head(x_pool)
        y_technique = self.technique_head(x_pool)
        
        return y_action, y_cell, y_number, y_technique