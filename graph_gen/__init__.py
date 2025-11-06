"""
Graph Generation Package for Smart Contract Vulnerability Detection
Provides AST, CFG, and DFG generation capabilities for Solidity contracts
"""

from .ast_generator import ASTGenerator
from .cfg_generator import CFGGenerator
from .dfg_generator import DFGGenerator
from .contract_data import ContractData, ContractDataset
from .main import GraphGenerationOrchestrator
from .solc_version_manager import SolcVersionManager

__version__ = "0.1.0"

__all__ = [
    "ASTGenerator",
    "CFGGenerator",
    "DFGGenerator",
    "ContractData",
    "ContractDataset",
    "GraphGenerationOrchestrator",
    "SolcVersionManager"
]
