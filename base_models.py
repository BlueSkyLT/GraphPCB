import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, BatchNorm, GCNConv, GATConv, SAGEConv


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3, dropout=0.0, use_batchnorm=True, use_bias=True):
        """
        Multi-layer GCN with configurable layers, dropout, batch norm, and bias.

        Args:
            in_dim (int): Input feature dimension.
            hidden_dim (int): Hidden layer dimension.
            out_dim (int): Output feature dimension (number of classes).
            num_layers (int): Number of GCN layers.
            dropout (float): Dropout rate.
            use_batchnorm (bool): Whether to use Batch Normalization.
            use_bias (bool): Whether to include bias in layers.
        """
        super(GCN, self).__init__()
        self.use_batchnorm = use_batchnorm
        self.num_layers = num_layers

        # GCN Layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList() if use_batchnorm else None

        # First layer (Input → Hidden)
        self.convs.append(GCNConv(in_dim, hidden_dim, bias=use_bias))
        if use_batchnorm:
            self.bns.append(BatchNorm(hidden_dim, allow_single_element=True))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim, bias=use_bias))
            if use_batchnorm:
                self.bns.append(BatchNorm(hidden_dim, allow_single_element=True))

        # Output layer (Hidden → Output)
        self.convs.append(GCNConv(hidden_dim, out_dim, bias=use_bias))

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.use_batchnorm and i < len(self.bns):
                x = self.bns[i](x)
            x = F.relu(x)
            x = self.dropout(x)

        return x


class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3, num_heads=4, dropout=0.0, use_batchnorm=True, use_bias=True):
        """
        Multi-layer GAT with configurable layers, dropout, batch norm, and bias.

        Args:
            in_dim (int): Input feature dimension.
            hidden_dim (int): Hidden layer dimension.
            out_dim (int): Output feature dimension (number of classes).
            num_layers (int): Number of GAT layers.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
            use_batchnorm (bool): Whether to use Batch Normalization.
            use_bias (bool): Whether to include bias in layers.
        """
        super(GAT, self).__init__()
        self.use_batchnorm = use_batchnorm
        self.num_layers = num_layers

        # GAT Layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList() if use_batchnorm else None

        # First layer (Input → Hidden)
        self.convs.append(GATConv(in_dim, hidden_dim, heads=num_heads, concat=True, bias=use_bias))
        if use_batchnorm:
            self.bns.append(BatchNorm(hidden_dim * num_heads))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, concat=True, bias=use_bias))
            if use_batchnorm:
                self.bns.append(BatchNorm(hidden_dim * num_heads))

        # Output layer (Hidden → Output)
        self.convs.append(GATConv(hidden_dim * num_heads, out_dim, heads=1, concat=False, bias=use_bias))
        if use_batchnorm:
            self.bns.append(BatchNorm(out_dim))

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.use_batchnorm and i < len(self.bns):
                x = self.bns[i](x)
            x = F.elu(x)  # ELU activation for GAT
            x = self.dropout(x)
        return x

class GIN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3, dropout=0.5, use_batchnorm=True):
        """
        Implements a GIN model with multiple layers and optional batch normalization.

        Args:
            in_dim (int): Input feature dimension.
            hidden_dim (int): Hidden layer dimension.
            out_dim (int): Output feature dimension (number of classes).
            num_layers (int): Number of GIN layers.
            dropout (float): Dropout rate.
            use_batchnorm (bool): Whether to use Batch Normalization.
        """
        super(GIN, self).__init__()
        self.use_batchnorm = use_batchnorm
        self.num_layers = num_layers

        # GIN Layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList() if use_batchnorm else None

        # First layer (Input → Hidden)
        mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.convs.append(GINConv(nn=mlp))
        if use_batchnorm:
            self.bns.append(BatchNorm(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            self.convs.append(GINConv(nn=mlp))
            if use_batchnorm:
                self.bns.append(BatchNorm(hidden_dim))

        # Output layer (Hidden → Output)
        mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim))
        self.convs.append(GINConv(nn=mlp))

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.use_batchnorm and i < len(self.bns):
                x = self.bns[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        return x


class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3, dropout=0.5,
                 use_batchnorm=True, aggr="mean", use_skip=True):
        """
        GraphSAGE with optional batch norm, dropout, and skip connections.
        """
        super(GraphSAGE, self).__init__()
        self.use_batchnorm = use_batchnorm
        self.use_skip = use_skip
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList() if use_batchnorm else None
        self.skip_projs = nn.ModuleList() if use_skip else None

        # First layer (input → hidden)
        self.convs.append(SAGEConv(in_dim, hidden_dim, aggr=aggr))
        if use_batchnorm:
            self.bns.append(BatchNorm(hidden_dim, allow_single_element=True))
        if use_skip:
            self.skip_projs.append(nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity())

        # Hidden layers (hidden → hidden)
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggr))
            if use_batchnorm:
                self.bns.append(BatchNorm(hidden_dim, allow_single_element=True))
            if use_skip:
                self.skip_projs.append(nn.Identity())  # same dim, no projection needed

        # Final layer (hidden → output)
        self.convs.append(SAGEConv(hidden_dim, out_dim, aggr=aggr))  # no skip/bn for final layer

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        residual = x  # For first skip connection

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)

            if self.use_batchnorm and i < len(self.bns):
                x = self.bns[i](x)

            if i < self.num_layers - 1:
                if self.use_skip:
                    skip = self.skip_projs[i](residual)
                    x = x + skip  # or replace with torch.cat([...], dim=1) for concat
                    residual = x  # update for next layer

                x = F.relu(x)
                x = self.dropout(x)
        return x

# This is a simple MLP model with optional batch normalization and residual connections.
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5, use_batchnorm=True, use_skip=True):
        """
        Multi-Layer Perceptron (MLP) with optional batch norm and residual (skip) connections.

        Args:
            input_dim (int): Input feature dimension.
            hidden_dim (int): Hidden layer dimension.
            output_dim (int): Output dimension (number of classes).
            num_layers (int): Total number of layers.
            dropout (float): Dropout rate.
            use_batchnorm (bool): Whether to apply batch normalization.
            use_skip (bool): Whether to use residual (skip) connections.
        """
        super(MLP, self).__init__()
        self.use_batchnorm = use_batchnorm
        self.use_skip = use_skip
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList() if use_batchnorm else None

        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        if use_batchnorm:
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_batchnorm:
                self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Output layer
        self.layers.append(nn.Linear(hidden_dim, output_dim))

        # Optional projection layer if input_dim ≠ hidden_dim
        self.proj = nn.Linear(input_dim, hidden_dim) if (input_dim != hidden_dim and use_skip) else None

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index=None):  # edge_index is ignored for MLP
        residual = x

        for i, layer in enumerate(self.layers):
            x = layer(x)

            if self.use_batchnorm and i < len(self.bns):
                x = self.bns[i](x)

            if i < self.num_layers - 1:
                # Residual connection
                if self.use_skip:
                    if i == 0:
                        res = residual if self.proj is None else self.proj(residual)
                    else:
                        res = residual
                    x = x + res
                    residual = x  # update for next layer

                x = F.relu(x)
                x = self.dropout(x)


        return x
