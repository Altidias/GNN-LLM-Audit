"""
TreeLSTM Module for AST Processing
Implements Child-Sum TreeLSTM for bottom-up hierarchical code structure encoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from typing import Dict, List, Tuple, Optional


class TreeLSTMCell(nn.Module):
    """
    Child-Sum TreeLSTM cell with separate forget gates for each child
    Aggregates information from variable number of children in AST
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Args:
            input_dim: Dimension of input node features
            hidden_dim: Dimension of hidden state and cell state
        """
        super(TreeLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim

        # Input gate
        self.W_i = nn.Linear(input_dim, hidden_dim)
        self.U_i = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Output gate
        self.W_o = nn.Linear(input_dim, hidden_dim)
        self.U_o = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Cell update
        self.W_u = nn.Linear(input_dim, hidden_dim)
        self.U_u = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Forget gate (applied per child)
        self.W_f = nn.Linear(input_dim, hidden_dim)
        self.U_f = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        child_h: List[torch.Tensor],
        child_c: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for TreeLSTM cell

        Args:
            x: Input features for current node [input_dim]
            child_h: List of child hidden states, each [hidden_dim]
            child_c: List of child cell states, each [hidden_dim]

        Returns:
            h: Hidden state [hidden_dim]
            c: Cell state [hidden_dim]
        """
        # Sum of child hidden states
        if len(child_h) == 0:
            child_h_sum = torch.zeros(self.hidden_dim, device=x.device)
        else:
            child_h_sum = torch.stack(child_h).sum(dim=0)

        # Input gate
        i = torch.sigmoid(self.W_i(x) + self.U_i(child_h_sum))

        # Output gate
        o = torch.sigmoid(self.W_o(x) + self.U_o(child_h_sum))

        # Cell update candidate
        u = torch.tanh(self.W_u(x) + self.U_u(child_h_sum))

        # Forget gates for each child
        if len(child_c) == 0:
            c = i * u
        else:
            f_list = []
            for h_k in child_h:
                f_k = torch.sigmoid(self.W_f(x) + self.U_f(h_k))
                f_list.append(f_k)

            # Combine child cells with forget gates
            c = i * u
            for f_k, c_k in zip(f_list, child_c):
                c = c + f_k * c_k

        # Hidden state
        h = o * torch.tanh(c)

        return h, c


class TreeLSTM(nn.Module):
    """
    TreeLSTM network for processing AST graphs
    Produces vulnerability-relevant embeddings from hierarchical code structure
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: Dimension of input node features
            hidden_dim: Dimension of hidden state
            output_dim: Dimension of output embeddings
            num_layers: Number of TreeLSTM layers
            dropout: Dropout probability
        """
        super(TreeLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # TreeLSTM layers
        self.tree_lstm_cells = nn.ModuleList([
            TreeLSTMCell(hidden_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        # Attention mechanism for node importance
        self.attention_proj = nn.Linear(hidden_dim, 1)

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through TreeLSTM

        Args:
            data: PyTorch Geometric Data object with:
                - x: Node features [num_nodes, input_dim]
                - edge_index: Edge connectivity [2, num_edges]
                - batch: Batch assignment [num_nodes]

        Returns:
            graph_embedding: Graph-level embedding [batch_size, output_dim]
            node_embeddings: Node-level embeddings [num_nodes, hidden_dim]
            attention_weights: Attention weights for each node [num_nodes, 1]
        """
        x = data.x
        edge_index = data.edge_index
        num_nodes = x.size(0)

        # Project input features
        x = self.input_proj(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Build adjacency structure for tree traversal
        # Convert edge_index to parent-child relationships
        parent_to_children = self._build_tree_structure(edge_index, num_nodes)

        # Topological sort for bottom-up traversal
        node_order = self._topological_sort(edge_index, num_nodes)

        # Process through TreeLSTM layers
        h_all = x
        for layer_idx in range(self.num_layers):
            h_new = torch.zeros(num_nodes, self.hidden_dim, device=x.device)
            c_new = torch.zeros(num_nodes, self.hidden_dim, device=x.device)

            # Bottom-up traversal
            for node_idx in node_order:
                children = parent_to_children[node_idx]

                # Gather child hidden and cell states
                child_h = [h_new[child] for child in children]
                child_c = [c_new[child] for child in children]

                # Apply TreeLSTM cell
                h, c = self.tree_lstm_cells[layer_idx](
                    h_all[node_idx],
                    child_h,
                    child_c
                )

                h_new[node_idx] = h
                c_new[node_idx] = c

            h_all = h_new
            h_all = self.dropout(h_all)

        # Node embeddings
        node_embeddings = h_all

        # Compute attention weights for each node
        attention_logits = self.attention_proj(node_embeddings)
        attention_weights = torch.softmax(attention_logits, dim=0)

        # Graph-level embedding via attention-weighted pooling
        if hasattr(data, 'batch'):
            # Batch processing
            batch = data.batch
            batch_size = batch.max().item() + 1
            graph_embeddings = []

            for i in range(batch_size):
                mask = (batch == i)
                node_emb = node_embeddings[mask]
                attn_weights = attention_weights[mask]

                # Weighted sum
                graph_emb = (node_emb * attn_weights).sum(dim=0)
                graph_embeddings.append(graph_emb)

            graph_embedding = torch.stack(graph_embeddings)
        else:
            # Single graph
            graph_embedding = (node_embeddings * attention_weights).sum(dim=0, keepdim=True)

        # Project to output dimension
        graph_embedding = self.output_proj(graph_embedding)

        return graph_embedding, node_embeddings, attention_weights

    def _build_tree_structure(
        self,
        edge_index: torch.Tensor,
        num_nodes: int
    ) -> Dict[int, List[int]]:
        """
        Build parent-to-children mapping from edge_index

        Args:
            edge_index: Edge connectivity [2, num_edges]
            num_nodes: Total number of nodes

        Returns:
            Dictionary mapping parent node to list of children
        """
        parent_to_children = {i: [] for i in range(num_nodes)}

        # Assuming edges go from parent to child
        for i in range(edge_index.size(1)):
            parent = edge_index[0, i].item()
            child = edge_index[1, i].item()
            parent_to_children[parent].append(child)

        return parent_to_children

    def _topological_sort(
        self,
        edge_index: torch.Tensor,
        num_nodes: int
    ) -> List[int]:
        """
        Topological sort for bottom-up tree traversal

        Args:
            edge_index: Edge connectivity [2, num_edges]
            num_nodes: Total number of nodes

        Returns:
            List of node indices in bottom-up order (leaves to root)
        """
        # Calculate in-degree for each node
        in_degree = torch.zeros(num_nodes, dtype=torch.long)
        for i in range(edge_index.size(1)):
            child = edge_index[1, i].item()
            in_degree[child] += 1

        # Find leaves (nodes with no children)
        queue = [i for i in range(num_nodes) if in_degree[i] == 0]
        result = []

        # Build child-to-parent mapping
        child_to_parent = {}
        for i in range(edge_index.size(1)):
            parent = edge_index[0, i].item()
            child = edge_index[1, i].item()
            if child not in child_to_parent:
                child_to_parent[child] = []
            child_to_parent[child].append(parent)

        # Process queue
        while queue:
            node = queue.pop(0)
            result.append(node)

            # Process parents
            if node in child_to_parent:
                for parent in child_to_parent[node]:
                    in_degree[parent] -= 1
                    if in_degree[parent] == 0:
                        queue.append(parent)

        return result
