# GNN Models for Smart Contract Analysis

Three specialized graph neural networks for processing contract graph representations: AST, CFG, and DFG.

## Models

**TreeLSTM** - Processes Abstract Syntax Trees using Child-Sum TreeLSTM architecture with bottom-up aggregation.

**RGCN** - Relational Graph Convolutional Network for Control Flow Graphs.

**EGNN** - Edge-Featured Graph Neural Network for Data Flow Graphs.

## Configuration

All models output 256-dimensional embeddings and include attention mechanisms for interpretability.

- TreeLSTM: input_dim=64, hidden_dim=256, output_dim=256, num_layers=2, dropout=0.1
- RGCN: input_dim=64, hidden_dim=256, output_dim=256, num_relations=6, num_layers=3, dropout=0.1
- EGNN: node_input_dim=64, edge_input_dim=16, hidden_dim=256, output_dim=256, num_layers=3, dropout=0.1

## Usage

```python
from graph_gen import ContractData
from gnn_models import TreeLSTM, RGCN, EGNN, GraphConverter

# Load contract with graphs
contract_data = ContractData.load("contract.pkl")

# Convert to PyTorch Geometric format
ast_data = GraphConverter.convert_ast(contract_data.get_ast())
cfg_data = GraphConverter.convert_cfg(contract_data.get_cfg())
dfg_data = GraphConverter.convert_dfg(contract_data.get_dfg())

# Initialize models
tree_lstm = TreeLSTM(input_dim=64)
rgcn = RGCN(input_dim=64, num_relations=6)
egnn = EGNN(node_input_dim=64, edge_input_dim=16)

# Generate embeddings
ast_embedding, _, _ = tree_lstm(ast_data)
cfg_embedding, _, _, _ = rgcn(cfg_data)
dfg_embedding, _, _, _, _, _ = egnn(dfg_data)
```

## Feature Encoding

**AST Node Features (64 dims):** Node type encoding plus attributes like visibility, is_constant, payable, view, pure.

**CFG Node Features (64 dims):** Node type encoding plus cfg_node_type attribute. Edge types encoded separately for RGCN relation-specific processing.

**DFG Node Features (64 dims):** Variable properties (var_kind, var_type, is_storage) and operation types (BinaryOperation, UnaryOperation with operation attributes like "BinaryType.ADDITION").

**DFG Edge Features (16 dims):** One-hot encoding of edge types: assignment, operand, result, parameter, read_by, writes_to, declares, return_value. Operations are stored on nodes rather than edges.

## Architecture

TreeLSTM uses topological sorting for bottom-up traversal and attention-weighted pooling for graph-level embeddings. RGCN applies different weight matrices per edge type with optional basis decomposition. EGNN treats edge features as first-class citizens and updates both node and edge representations jointly. All models use residual connections, batch normalization, and dropout.
