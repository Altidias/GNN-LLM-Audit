"""
Graph Converter for NetworkX to PyTorch Geometric
Converts generated graphs to PyTorch Geometric Data objects
"""

import torch
import networkx as nx
from torch_geometric.data import Data
from typing import Dict, List, Tuple, Optional
import numpy as np


class GraphConverter:
    """
    Converts NetworkX graphs from graph generators to PyTorch Geometric Data objects
    Handles node/edge feature extraction and encoding
    """

    @staticmethod
    def networkx_to_pyg(
        G: nx.DiGraph,
        node_feature_keys: Optional[List[str]] = None,
        edge_feature_keys: Optional[List[str]] = None,
        node_feature_dim: int = 64,
        edge_feature_dim: int = 16
    ) -> Data:
        """
        Convert NetworkX graph to PyTorch Geometric Data

        Args:
            G: NetworkX directed graph
            node_feature_keys: List of node attribute keys to use as features
            edge_feature_keys: List of edge attribute keys to use as features
            node_feature_dim: Dimension of node feature vectors
            edge_feature_dim: Dimension of edge feature vectors

        Returns:
            PyTorch Geometric Data object
        """
        # Create node mapping
        node_mapping = {node: idx for idx, node in enumerate(G.nodes())}
        num_nodes = len(node_mapping)

        # Extract node features
        node_features = GraphConverter._extract_node_features(
            G, node_mapping, node_feature_keys, node_feature_dim
        )

        # Extract edge information
        edge_index, edge_features, edge_types = GraphConverter._extract_edge_features(
            G, node_mapping, edge_feature_keys, edge_feature_dim
        )

        # Create PyTorch Geometric Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_features,
            num_nodes=num_nodes
        )

        # Add edge types if available
        if edge_types is not None:
            data.edge_type = edge_types

        return data

    @staticmethod
    def _extract_node_features(
        G: nx.DiGraph,
        node_mapping: Dict,
        feature_keys: Optional[List[str]],
        feature_dim: int
    ) -> torch.Tensor:
        """
        Extract node features from NetworkX graph

        Args:
            G: NetworkX graph
            node_mapping: Mapping from node IDs to indices
            feature_keys: Keys to extract from node attributes
            feature_dim: Dimension of feature vector

        Returns:
            Node feature matrix [num_nodes, feature_dim]
        """
        num_nodes = len(node_mapping)
        node_features = torch.zeros(num_nodes, feature_dim)

        for node, idx in node_mapping.items():
            node_data = G.nodes[node]

            # Encode node type (one-hot in first dimensions)
            node_type = node_data.get('node_type', 'unknown')
            type_encoding = GraphConverter._encode_node_type(node_type)
            node_features[idx, :len(type_encoding)] = type_encoding

            # Add additional features if specified
            if feature_keys:
                offset = len(type_encoding)
                for key in feature_keys:
                    if key in node_data and offset < feature_dim:
                        value = node_data[key]
                        # Convert to numeric if possible
                        if isinstance(value, (int, float)):
                            node_features[idx, offset] = float(value)
                            offset += 1
                        elif isinstance(value, bool):
                            node_features[idx, offset] = 1.0 if value else 0.0
                            offset += 1

        return node_features

    @staticmethod
    def _extract_edge_features(
        G: nx.DiGraph,
        node_mapping: Dict,
        feature_keys: Optional[List[str]],
        feature_dim: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Extract edge information from NetworkX graph

        Args:
            G: NetworkX graph
            node_mapping: Mapping from node IDs to indices
            feature_keys: Keys to extract from edge attributes
            feature_dim: Dimension of edge feature vector

        Returns:
            edge_index: Edge connectivity [2, num_edges]
            edge_features: Edge feature matrix [num_edges, feature_dim]
            edge_types: Edge type indices [num_edges] (if applicable)
        """
        edges = list(G.edges(data=True))
        num_edges = len(edges)

        edge_index = torch.zeros(2, num_edges, dtype=torch.long)
        edge_features = torch.zeros(num_edges, feature_dim)
        edge_types = []

        for i, (src, dst, edge_data) in enumerate(edges):
            # Edge connectivity
            edge_index[0, i] = node_mapping[src]
            edge_index[1, i] = node_mapping[dst]

            # Encode edge type
            edge_type = edge_data.get('edge_type', 'unknown')
            type_encoding = GraphConverter._encode_edge_type(edge_type, edge_data)
            edge_features[i, :len(type_encoding)] = type_encoding

            # Store edge type index for RGCN
            edge_type_idx = GraphConverter._get_edge_type_index(edge_type, edge_data)
            edge_types.append(edge_type_idx)

            # Add additional features if specified
            if feature_keys:
                offset = len(type_encoding)
                for key in feature_keys:
                    if key in edge_data and offset < feature_dim:
                        value = edge_data[key]
                        if isinstance(value, (int, float)):
                            edge_features[i, offset] = float(value)
                            offset += 1
                        elif isinstance(value, bool):
                            edge_features[i, offset] = 1.0 if value else 0.0
                            offset += 1

        edge_types_tensor = torch.tensor(edge_types, dtype=torch.long) if edge_types else None

        return edge_index, edge_features, edge_types_tensor

    @staticmethod
    def _encode_node_type(node_type: str) -> torch.Tensor:
        """
        One-hot encode node type

        Args:
            node_type: String node type

        Returns:
            One-hot encoding vector
        """
        # Common node types
        type_map = {
            'Contract': 0, 'Function': 1, 'Variable': 2, 'StateVariable': 3,
            'LocalVariable': 4, 'Parameter': 5, 'ReturnParameter': 6,
            'Modifier': 7, 'Event': 8, 'Expression': 9, 'Conditional': 10,
            'Return': 11, 'Loop': 12, 'FunctionCall': 13, 'Operation': 14,
            'BinaryOperation': 15, 'UnaryOperation': 16, 'Constant': 17
        }

        encoding = torch.zeros(20)  # Reserve 20 dimensions for types
        if node_type in type_map:
            encoding[type_map[node_type]] = 1.0
        else:
            encoding[19] = 1.0  # Unknown type

        return encoding

    @staticmethod
    def _encode_edge_type(edge_type: str, edge_data: Dict) -> torch.Tensor:
        """
        Encode edge type and attributes

        Args:
            edge_type: String edge type
            edge_data: Edge attribute dictionary

        Returns:
            Feature encoding vector
        """
        encoding = torch.zeros(10)

        # Edge type encoding
        type_map = {
            'control_flow': 0, 'sequential': 0,
            'contains': 1, 'has_function': 1, 'has_state_var': 1,
            'inherits': 2,
            'calls': 3,
            'assignment': 4,
            'operand': 5,
            'result': 6,
            'argument': 7,
            'returns': 8
        }

        if edge_type in type_map:
            encoding[type_map[edge_type]] = 1.0
        else:
            encoding[9] = 1.0  # Unknown

        # Additional attributes
        if edge_data.get('branch') == 'true':
            encoding[9] = 0.5
        elif edge_data.get('branch') == 'false':
            encoding[9] = -0.5

        return encoding

    @staticmethod
    def _get_edge_type_index(edge_type: str, edge_data: Dict) -> int:
        """
        Get edge type index for RGCN

        Args:
            edge_type: String edge type
            edge_data: Edge attribute dictionary

        Returns:
            Edge type index
        """
        # Map to RGCN edge type constants
        if edge_type in ['control_flow', 'sequential', 'entry']:
            return 0  # EDGE_SEQUENTIAL
        elif edge_data.get('branch') == 'true':
            return 1  # EDGE_CONDITIONAL_TRUE
        elif edge_data.get('branch') == 'false':
            return 2  # EDGE_CONDITIONAL_FALSE
        elif 'loop' in edge_type.lower():
            return 3  # EDGE_LOOP
        elif 'call' in edge_type.lower():
            return 4  # EDGE_CALL
        elif 'return' in edge_type.lower():
            return 5  # EDGE_RETURN
        else:
            return 0  # Default to sequential

    @staticmethod
    def convert_ast(ast_graph: nx.DiGraph) -> Data:
        """
        Convert AST graph to PyTorch Geometric format

        Args:
            ast_graph: NetworkX AST graph

        Returns:
            PyTorch Geometric Data object
        """
        return GraphConverter.networkx_to_pyg(
            ast_graph,
            node_feature_keys=['visibility', 'is_constant', 'payable', 'view', 'pure'],
            node_feature_dim=64
        )

    @staticmethod
    def convert_cfg(cfg_graph: nx.DiGraph) -> Data:
        """
        Convert CFG graph to PyTorch Geometric format

        Args:
            cfg_graph: NetworkX CFG graph

        Returns:
            PyTorch Geometric Data object with edge types
        """
        return GraphConverter.networkx_to_pyg(
            cfg_graph,
            node_feature_keys=['cfg_node_type'],
            edge_feature_keys=['branch'],
            node_feature_dim=64,
            edge_feature_dim=16
        )

    @staticmethod
    def convert_dfg(dfg_graph: nx.DiGraph) -> Data:
        """
        Convert DFG graph to PyTorch Geometric format

        Args:
            dfg_graph: NetworkX DFG graph

        Returns:
            PyTorch Geometric Data object with edge type features
        """
        return GraphConverter.networkx_to_pyg(
            dfg_graph,
            node_feature_keys=['var_kind', 'var_type', 'is_storage', 'node_type', 'operation'],
            edge_feature_keys=['edge_type'],
            node_feature_dim=64,
            edge_feature_dim=16  # Edge types only (operations stored on nodes)
        )
