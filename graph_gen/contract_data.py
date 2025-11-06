"""
Contract Data Storage Format
Unified format for storing Solidity contracts with their graph representations
Supports serialization/deserialization with pickle for efficient storage
"""

import pickle
import json
import networkx as nx
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime


class ContractData:
    """
    Container for a smart contract and its associated graph representations.

    Stores:
    - Original Solidity source code
    - AST (Abstract Syntax Tree) graph
    - CFG (Control Flow Graph) graph
    - DFG (Data Flow Graph) graph
    - Metadata about the contract and generation
    """

    def __init__(
        self,
        contract_source: str,
        contract_name: str,
        contract_path: str,
        ast_graph: Optional[nx.DiGraph] = None,
        cfg_graph: Optional[nx.DiGraph] = None,
        dfg_graph: Optional[nx.DiGraph] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize ContractData

        Args:
            contract_source: Full Solidity source code
            contract_name: Name of the contract
            contract_path: Original file path
            ast_graph: AST graph representation
            cfg_graph: CFG graph representation
            dfg_graph: DFG graph representation
            metadata: Additional metadata
        """
        self.contract_source = contract_source
        self.contract_name = contract_name
        self.contract_path = contract_path
        self.ast_graph = ast_graph
        self.cfg_graph = cfg_graph
        self.dfg_graph = dfg_graph
        self.metadata = metadata or {}
        self.generation_timestamp = datetime.now().isoformat()

    def set_ast(self, ast_graph: nx.DiGraph):
        """Set the AST graph"""
        self.ast_graph = ast_graph

    def set_cfg(self, cfg_graph: nx.DiGraph):
        """Set the CFG graph"""
        self.cfg_graph = cfg_graph

    def set_dfg(self, dfg_graph: nx.DiGraph):
        """Set the DFG graph"""
        self.dfg_graph = dfg_graph

    def get_ast(self) -> Optional[nx.DiGraph]:
        """Get the AST graph"""
        return self.ast_graph

    def get_cfg(self) -> Optional[nx.DiGraph]:
        """Get the CFG graph"""
        return self.cfg_graph

    def get_dfg(self) -> Optional[nx.DiGraph]:
        """Get the DFG graph"""
        return self.dfg_graph

    def get_source(self) -> str:
        """Get the contract source code"""
        return self.contract_source

    def has_all_graphs(self) -> bool:
        """Check if all three graphs have been generated"""
        return (
            self.ast_graph is not None
            and self.cfg_graph is not None
            and self.dfg_graph is not None
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the contract data"""
        summary = {
            "contract_name": self.contract_name,
            "contract_path": self.contract_path,
            "source_lines": len(self.contract_source.splitlines()),
            "generation_timestamp": self.generation_timestamp,
            "has_ast": self.ast_graph is not None,
            "has_cfg": self.cfg_graph is not None,
            "has_dfg": self.dfg_graph is not None,
        }

        # Add graph statistics if available
        if self.ast_graph:
            summary["ast_nodes"] = self.ast_graph.number_of_nodes()
            summary["ast_edges"] = self.ast_graph.number_of_edges()

        if self.cfg_graph:
            summary["cfg_nodes"] = self.cfg_graph.number_of_nodes()
            summary["cfg_edges"] = self.cfg_graph.number_of_edges()

        if self.dfg_graph:
            summary["dfg_nodes"] = self.dfg_graph.number_of_nodes()
            summary["dfg_edges"] = self.dfg_graph.number_of_edges()

        summary.update(self.metadata)

        return summary

    def save(self, output_path: str):
        """
        Save ContractData to a pickle file

        Args:
            output_path: Path to save the pickle file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def save_graphs_as_json(self, output_dir: str):
        """
        Save individual graphs as JSON files (for human readability/debugging)

        Args:
            output_dir: Directory to save JSON files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        base_name = f"{self.contract_name}"

        # Save AST
        if self.ast_graph:
            ast_data = nx.node_link_data(self.ast_graph)
            with open(output_dir / f"{base_name}_ast.json", 'w') as f:
                json.dump(ast_data, f, indent=2)

        # Save CFG
        if self.cfg_graph:
            cfg_data = nx.node_link_data(self.cfg_graph)
            with open(output_dir / f"{base_name}_cfg.json", 'w') as f:
                json.dump(cfg_data, f, indent=2)

        # Save DFG
        if self.dfg_graph:
            dfg_data = nx.node_link_data(self.dfg_graph)
            with open(output_dir / f"{base_name}_dfg.json", 'w') as f:
                json.dump(dfg_data, f, indent=2)

        # Save summary
        with open(output_dir / f"{base_name}_summary.json", 'w') as f:
            json.dump(self.get_summary(), f, indent=2)

    @staticmethod
    def load(input_path: str) -> 'ContractData':
        """
        Load ContractData from a pickle file

        Args:
            input_path: Path to the pickle file

        Returns:
            ContractData object
        """
        with open(input_path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def load_from_json(json_dir: str, contract_name: str) -> 'ContractData':
        """
        Load ContractData from JSON files

        Args:
            json_dir: Directory containing JSON files
            contract_name: Name of the contract

        Returns:
            ContractData object (without source code)
        """
        json_dir = Path(json_dir)
        base_name = contract_name

        # Load summary
        with open(json_dir / f"{base_name}_summary.json", 'r') as f:
            summary = json.load(f)

        contract_data = ContractData(
            contract_source="",  # Source not stored in JSON
            contract_name=contract_name,
            contract_path=summary.get("contract_path", ""),
            metadata=summary
        )

        # Load AST
        ast_path = json_dir / f"{base_name}_ast.json"
        if ast_path.exists():
            with open(ast_path, 'r') as f:
                ast_data = json.load(f)
                contract_data.ast_graph = nx.node_link_graph(ast_data)

        # Load CFG
        cfg_path = json_dir / f"{base_name}_cfg.json"
        if cfg_path.exists():
            with open(cfg_path, 'r') as f:
                cfg_data = json.load(f)
                contract_data.cfg_graph = nx.node_link_graph(cfg_data)

        # Load DFG
        dfg_path = json_dir / f"{base_name}_dfg.json"
        if dfg_path.exists():
            with open(dfg_path, 'r') as f:
                dfg_data = json.load(f)
                contract_data.dfg_graph = nx.node_link_graph(dfg_data)

        return contract_data

    def __repr__(self) -> str:
        """String representation"""
        return (
            f"ContractData(name='{self.contract_name}', "
            f"has_ast={self.ast_graph is not None}, "
            f"has_cfg={self.cfg_graph is not None}, "
            f"has_dfg={self.dfg_graph is not None})"
        )


class ContractDataset:
    """
    Collection of ContractData objects for batch processing
    """

    def __init__(self):
        self.contracts: Dict[str, ContractData] = {}

    def add_contract(self, contract_data: ContractData):
        """Add a contract to the dataset"""
        self.contracts[contract_data.contract_name] = contract_data

    def get_contract(self, contract_name: str) -> Optional[ContractData]:
        """Get a contract by name"""
        return self.contracts.get(contract_name)

    def get_all_contracts(self) -> Dict[str, ContractData]:
        """Get all contracts"""
        return self.contracts

    def save(self, output_path: str):
        """Save the entire dataset"""
        with open(output_path, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(input_path: str) -> 'ContractDataset':
        """Load a dataset"""
        with open(input_path, 'rb') as f:
            return pickle.load(f)

    def __len__(self) -> int:
        """Number of contracts in dataset"""
        return len(self.contracts)

    def __repr__(self) -> str:
        """String representation"""
        return f"ContractDataset(contracts={len(self.contracts)})"
