"""
Relational Graph Convolutional Network (RGCN) for CFG Processing
Implements relation-specific transformations for different control flow edge types
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from typing import Dict, Tuple, Optional


class RGCNConv(MessagePassing):
    """
    Relational GCN layer with separate weight matrices for each edge type
    Handles different control flow patterns (sequential, conditional, loops, calls)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_relations: int,
        num_bases: Optional[int] = None,
        aggr: str = 'mean'
    ):
        """
        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension
            num_relations: Number of different edge types
            num_bases: Number of basis matrices for basis decomposition (None = use full matrices)
            aggr: Aggregation method ('mean', 'sum', 'max')
        """
        super(RGCNConv, self).__init__(aggr=aggr)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases

        if num_bases is not None:
            # Basis decomposition for parameter efficiency
            self.basis = nn.Parameter(torch.Tensor(num_bases, in_channels, out_channels))
            self.att = nn.Parameter(torch.Tensor(num_relations, num_bases))
        else:
            # Full weight matrix per relation
            self.weight = nn.Parameter(torch.Tensor(num_relations, in_channels, out_channels))

        # Self-loop weight
        self.root = nn.Parameter(torch.Tensor(in_channels, out_channels))

        # Bias
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters"""
        if self.num_bases is not None:
            nn.init.xavier_uniform_(self.basis)
            nn.init.xavier_uniform_(self.att)
        else:
            nn.init.xavier_uniform_(self.weight)

        nn.init.xavier_uniform_(self.root)
        nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
            edge_type: Edge type for each edge [num_edges]

        Returns:
            Updated node features [num_nodes, out_channels]
        """
        # Self-loop for root transformation
        out = torch.matmul(x, self.root)

        # Process each relation type
        for r in range(self.num_relations):
            # Get edges of this relation type
            mask = edge_type == r
            if mask.sum() == 0:
                continue

            edge_index_r = edge_index[:, mask]

            # Get weight matrix for this relation
            if self.num_bases is not None:
                # Basis decomposition
                w = torch.einsum('rb,bio->rio', self.att[r], self.basis)
            else:
                w = self.weight[r]

            # Message passing for this relation
            out = out + self.propagate(edge_index_r, x=x, weight=w)

        out = out + self.bias

        return out

    def message(self, x_j: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        Construct messages from neighbors

        Args:
            x_j: Neighbor node features [num_edges, in_channels]
            weight: Relation-specific weight matrix [in_channels, out_channels]

        Returns:
            Messages [num_edges, out_channels]
        """
        return torch.matmul(x_j, weight)


class RGCN(nn.Module):
    """
    Relational Graph Convolutional Network for CFG processing
    Handles different control flow edge types with relation-specific transformations
    """

    # Edge type constants
    EDGE_SEQUENTIAL = 0
    EDGE_CONDITIONAL_TRUE = 1
    EDGE_CONDITIONAL_FALSE = 2
    EDGE_LOOP = 3
    EDGE_CALL = 4
    EDGE_RETURN = 5

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_relations: int = 6,
        num_layers: int = 3,
        num_bases: Optional[int] = 30,
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: Dimension of input node features
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output embeddings
            num_relations: Number of edge types
            num_layers: Number of RGCN layers
            num_bases: Number of basis matrices (None for full matrices)
            dropout: Dropout probability
        """
        super(RGCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_relations = num_relations
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # RGCN layers
        self.convs = nn.ModuleList()
        self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations, num_bases))
        for _ in range(num_layers - 1):
            self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations, num_bases))

        # Batch normalization
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        # Attention mechanism for node importance
        self.node_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Edge attention for control flow pattern importance
        self.edge_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2 + num_relations, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through RGCN

        Args:
            data: PyTorch Geometric Data object with:
                - x: Node features [num_nodes, input_dim]
                - edge_index: Edge connectivity [2, num_edges]
                - edge_type: Edge type for each edge [num_edges]
                - batch: Batch assignment [num_nodes] (optional)

        Returns:
            graph_embedding: Graph-level embedding [batch_size, output_dim]
            node_embeddings: Node-level embeddings [num_nodes, hidden_dim]
            node_attention_weights: Attention weights for each node [num_nodes, 1]
            edge_attention_weights: Attention weights for each edge [num_edges, 1]
        """
        x = data.x
        edge_index = data.edge_index
        edge_type = data.edge_type

        # Project input features
        x = self.input_proj(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Apply RGCN layers
        for i, conv in enumerate(self.convs):
            x_new = conv(x, edge_index, edge_type)
            x_new = self.batch_norms[i](x_new)
            x_new = F.relu(x_new)
            x_new = self.dropout(x_new)

            # Residual connection (if dimensions match)
            if i > 0:
                x = x + x_new
            else:
                x = x_new

        # Node embeddings
        node_embeddings = x

        # Compute node attention weights
        node_attention_logits = self.node_attention(node_embeddings)
        node_attention_weights = torch.softmax(node_attention_logits, dim=0)

        # Compute edge attention weights
        edge_src = node_embeddings[edge_index[0]]
        edge_dst = node_embeddings[edge_index[1]]
        edge_type_onehot = F.one_hot(edge_type, num_classes=self.num_relations).float()
        edge_features = torch.cat([edge_src, edge_dst, edge_type_onehot], dim=-1)
        edge_attention_logits = self.edge_attention(edge_features)
        edge_attention_weights = torch.sigmoid(edge_attention_logits)

        # Graph-level embedding via attention-weighted pooling
        if hasattr(data, 'batch'):
            batch = data.batch
            batch_size = batch.max().item() + 1
            graph_embeddings = []

            for i in range(batch_size):
                mask = (batch == i)
                node_emb = node_embeddings[mask]
                attn_weights = node_attention_weights[mask]

                # Weighted sum
                graph_emb = (node_emb * attn_weights).sum(dim=0)
                graph_embeddings.append(graph_emb)

            graph_embedding = torch.stack(graph_embeddings)
        else:
            # Single graph
            graph_embedding = (node_embeddings * node_attention_weights).sum(dim=0, keepdim=True)

        # Project to output dimension
        graph_embedding = self.output_proj(graph_embedding)

        return graph_embedding, node_embeddings, node_attention_weights, edge_attention_weights

    @staticmethod
    def encode_edge_types(edge_attributes: Dict[str, any]) -> int:
        """
        Convert edge attributes to edge type index

        Args:
            edge_attributes: Dictionary of edge attributes

        Returns:
            Edge type index
        """
        edge_type = edge_attributes.get('edge_type', 'sequential')

        if edge_type == 'control_flow' or edge_type == 'sequential':
            return RGCN.EDGE_SEQUENTIAL
        elif edge_type == 'conditional' or edge_attributes.get('branch') == 'true':
            return RGCN.EDGE_CONDITIONAL_TRUE
        elif edge_attributes.get('branch') == 'false':
            return RGCN.EDGE_CONDITIONAL_FALSE
        elif 'loop' in edge_type.lower():
            return RGCN.EDGE_LOOP
        elif 'call' in edge_type.lower():
            return RGCN.EDGE_CALL
        elif 'return' in edge_type.lower():
            return RGCN.EDGE_RETURN
        else:
            return RGCN.EDGE_SEQUENTIAL
