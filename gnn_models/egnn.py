"""
Edge-Featured Graph Neural Network (EGNN) for DFG Processing
Treats edge features as first-class citizens in message passing
Incorporates data operations, taint status, and value transformations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from typing import Dict, Tuple, Optional


class EGNNConv(MessagePassing):
    """
    Edge-featured GNN layer that incorporates edge features into message passing
    Processes data flow operations and transformations
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        aggr: str = 'add'
    ):
        """
        Args:
            node_dim: Node feature dimension
            edge_dim: Edge feature dimension
            hidden_dim: Hidden dimension for message computation
            aggr: Aggregation method ('add', 'mean', 'max')
        """
        super(EGNNConv, self).__init__(aggr=aggr)
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim

        # Edge message network
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )

        # Node update network
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )

        # Edge update network
        self.edge_update_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with edge feature updates

        Args:
            x: Node features [num_nodes, node_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]

        Returns:
            x_updated: Updated node features [num_nodes, node_dim]
            edge_attr_updated: Updated edge features [num_edges, edge_dim]
        """
        # Update edge features first
        edge_attr_updated = self._update_edge_features(x, edge_index, edge_attr)

        # Message passing with updated edge features
        x_updated = self.propagate(edge_index, x=x, edge_attr=edge_attr_updated)

        # Update node features
        x_updated = self.node_mlp(torch.cat([x, x_updated], dim=-1))

        return x_updated, edge_attr_updated

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Construct messages incorporating edge features

        Args:
            x_i: Target node features [num_edges, node_dim]
            x_j: Source node features [num_edges, node_dim]
            edge_attr: Edge features [num_edges, edge_dim]

        Returns:
            Messages [num_edges, node_dim]
        """
        # Concatenate source, target, and edge features
        edge_input = torch.cat([x_i, x_j, edge_attr], dim=-1)

        # Compute message
        message = self.edge_mlp(edge_input)

        return message

    def _update_edge_features(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Update edge features based on connected nodes

        Args:
            x: Node features [num_nodes, node_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]

        Returns:
            Updated edge features [num_edges, edge_dim]
        """
        src_nodes = x[edge_index[0]]
        dst_nodes = x[edge_index[1]]

        # Concatenate node and edge features
        edge_input = torch.cat([src_nodes, dst_nodes, edge_attr], dim=-1)

        # Update edge features
        edge_attr_updated = self.edge_update_mlp(edge_input)

        return edge_attr_updated


class EGNN(nn.Module):
    """
    Edge-Featured Graph Neural Network for DFG processing
    Captures data operations, taint propagation, and value transformations
    """

    def __init__(
        self,
        node_input_dim: int,
        edge_input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        """
        Args:
            node_input_dim: Dimension of input node features
            edge_input_dim: Dimension of input edge features
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output embeddings
            num_layers: Number of EGNN layers
            dropout: Dropout probability
        """
        super(EGNN, self).__init__()
        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Input projections
        self.node_input_proj = nn.Linear(node_input_dim, hidden_dim)
        self.edge_input_proj = nn.Linear(edge_input_dim, hidden_dim)

        # EGNN layers
        self.convs = nn.ModuleList([
            EGNNConv(hidden_dim, hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        # Batch normalization
        self.node_batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        self.edge_batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        # Node attention mechanism
        self.node_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Edge attention mechanism (for identifying critical data flows)
        self.edge_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Taint propagation tracking
        self.taint_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        data: Data
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through EGNN

        Args:
            data: PyTorch Geometric Data object with:
                - x: Node features [num_nodes, node_input_dim]
                - edge_index: Edge connectivity [2, num_edges]
                - edge_attr: Edge features [num_edges, edge_input_dim]
                - batch: Batch assignment [num_nodes] (optional)

        Returns:
            graph_embedding: Graph-level embedding [batch_size, output_dim]
            node_embeddings: Node-level embeddings [num_nodes, hidden_dim]
            edge_embeddings: Edge-level embeddings [num_edges, hidden_dim]
            node_attention_weights: Attention weights for each node [num_nodes, 1]
            edge_attention_weights: Attention weights for each edge [num_edges, 1]
            taint_scores: Taint probability for each node [num_nodes, 1]
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # Project input features
        x = self.node_input_proj(x)
        x = F.relu(x)
        x = self.dropout(x)

        edge_attr = self.edge_input_proj(edge_attr)
        edge_attr = F.relu(edge_attr)
        edge_attr = self.dropout(edge_attr)

        # Apply EGNN layers
        for i, conv in enumerate(self.convs):
            x_new, edge_attr_new = conv(x, edge_index, edge_attr)

            # Batch normalization
            x_new = self.node_batch_norms[i](x_new)
            edge_attr_new = self.edge_batch_norms[i](edge_attr_new)

            # Activation
            x_new = F.relu(x_new)
            edge_attr_new = F.relu(edge_attr_new)

            # Dropout
            x_new = self.dropout(x_new)
            edge_attr_new = self.dropout(edge_attr_new)

            # Residual connections
            if i > 0:
                x = x + x_new
                edge_attr = edge_attr + edge_attr_new
            else:
                x = x_new
                edge_attr = edge_attr_new

        # Final embeddings
        node_embeddings = x
        edge_embeddings = edge_attr

        # Compute attention weights
        node_attention_logits = self.node_attention(node_embeddings)
        node_attention_weights = torch.softmax(node_attention_logits, dim=0)

        edge_attention_logits = self.edge_attention(edge_embeddings)
        edge_attention_weights = torch.sigmoid(edge_attention_logits)

        # Compute taint scores
        taint_scores = self.taint_predictor(node_embeddings)

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

        return (
            graph_embedding,
            node_embeddings,
            edge_embeddings,
            node_attention_weights,
            edge_attention_weights,
            taint_scores
        )

    @staticmethod
    def encode_edge_features(edge_attributes: Dict[str, any], feature_dim: int = 16) -> torch.Tensor:
        """
        Convert edge attributes to feature vector based on current DFG generation

        The current DFG generator stores operations on intermediate operation nodes
        rather than on edges, so edge features primarily encode relationship types.

        Encoding breakdown:
        - Dimensions 0-7: Edge types (one-hot encoding)
        - Dimensions 8-15: Reserved for future edge attributes

        Edge types in current DFG:
        - assignment: Direct variable assignment
        - operand: Edge from variable to operation node
        - result: Edge from operation node to result variable
        - parameter: Function parameter declaration
        - read_by: State variable read by function
        - writes_to: Function writes to state variable
        - declares: Contract/function declares variable
        - return_value: Function return value

        Args:
            edge_attributes: Dictionary of edge attributes
            feature_dim: Dimension of output feature vector (default 16)

        Returns:
            Edge feature vector encoding relationship type
        """
        features = torch.zeros(feature_dim)

        # Edge type encoding (dimensions 0-7, one-hot)
        edge_type = edge_attributes.get('edge_type', 'unknown')
        type_map = {
            'assignment': 0,
            'operand': 1,
            'result': 2,
            'parameter': 3,
            'read_by': 4,
            'writes_to': 5,
            'declares': 6,
            'return_value': 7
        }

        if edge_type in type_map:
            features[type_map[edge_type]] = 1.0

        # Note: operation attribute only appears on assignment edges with value "="
        # Rich operation information (ADDITION, SUBTRACTION, etc.) is stored on
        # BinaryOperation/UnaryOperation nodes, not on edges

        return features
