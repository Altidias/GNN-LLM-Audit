"""
GNN Models Package for Smart Contract Vulnerability Detection
Provides specialized graph neural networks for processing AST, CFG, and DFG representations
"""

from .tree_lstm import TreeLSTM, TreeLSTMCell
from .rgcn import RGCN, RGCNConv
from .egnn import EGNN, EGNNConv

__version__ = "0.1.0"

__all__ = [
    "TreeLSTM",
    "TreeLSTMCell",
    "RGCN",
    "RGCNConv",
    "EGNN",
    "EGNNConv"
]
