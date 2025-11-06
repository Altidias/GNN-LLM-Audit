"""
AST (Abstract Syntax Tree) Graph Generator for Solidity Smart Contracts
Uses Slither to extract and generate AST representations
"""

import networkx as nx
import os
import logging
from typing import Dict, Any, Optional
from slither import Slither
from slither.core.declarations import Contract, Function
from slither.core.cfg.node import Node
from solc_version_manager import SolcVersionManager

logger = logging.getLogger(__name__)


class ASTGenerator:
    """Generates Abstract Syntax Tree representation of smart contracts"""

    def __init__(self):
        self.graph = None

    def generate(self, contract_path: str, contract_name: Optional[str] = None, auto_install_solc: bool = True) -> nx.DiGraph:
        """
        Generate AST graph from a Solidity contract

        Args:
            contract_path: Path to the Solidity contract file
            contract_name: Specific contract name to analyze (optional)
            auto_install_solc: Automatically install and switch solc version (default: True)

        Returns:
            NetworkX DiGraph representing the AST
        """
        # Auto-select appropriate solc version based on pragma
        if auto_install_solc:
            logger.info("Auto-detecting and switching solc version...")
            version_switched = SolcVersionManager.auto_select_and_use_version(contract_path, auto_install=True)

            # Verify the switch
            current_solc = SolcVersionManager.get_current_version()
            if current_solc:
                logger.info(f"Using solc version: {current_solc}")
            else:
                logger.warning("Could not verify solc version")

        # Windows path fix: change to contract directory and use relative path
        original_dir = os.getcwd()
        contract_path = os.path.abspath(contract_path)
        contract_dir = os.path.dirname(contract_path)
        contract_file = os.path.basename(contract_path)

        logger.debug(f"Analyzing contract: {contract_file} in directory: {contract_dir}")

        try:
            os.chdir(contract_dir)
            logger.debug("Running Slither analysis...")
            slither = Slither(contract_file)
            logger.debug(f"Slither found {len(slither.contracts)} contract(s)")
        finally:
            os.chdir(original_dir)

        # Get the target contract
        if contract_name:
            contract = next((c for c in slither.contracts if c.name == contract_name), None)
            if not contract:
                raise ValueError(f"Contract '{contract_name}' not found in {contract_path}")
        else:
            # Use the first contract if no name specified
            contract = slither.contracts[0] if slither.contracts else None
            if not contract:
                raise ValueError(f"No contracts found in {contract_path}")

        self.graph = nx.DiGraph()
        self._build_ast(contract)

        return self.graph

    def _build_ast(self, contract: Contract):
        """Build AST from contract structure"""
        # Add contract node
        contract_id = f"contract_{contract.name}"
        self.graph.add_node(
            contract_id,
            node_type="Contract",
            name=contract.name,
            kind=contract.contract_kind,
            is_interface=contract.is_interface,
            is_library=contract.is_library
        )

        # Add inheritance relationships
        for base_contract in contract.inheritance:
            base_id = f"contract_{base_contract.name}"
            if not self.graph.has_node(base_id):
                self.graph.add_node(
                    base_id,
                    node_type="Contract",
                    name=base_contract.name,
                    kind=base_contract.contract_kind
                )
            self.graph.add_edge(contract_id, base_id, edge_type="inherits")

        # Add state variables
        for var in contract.state_variables:
            var_id = f"state_var_{contract.name}_{var.name}"
            self.graph.add_node(
                var_id,
                node_type="StateVariable",
                name=var.name,
                var_type=str(var.type),
                visibility=str(var.visibility),
                is_constant=var.is_constant,
                is_immutable=var.is_immutable
            )
            self.graph.add_edge(contract_id, var_id, edge_type="has_state_var")

        # Add functions
        for func in contract.functions:
            func_id = self._add_function_to_ast(contract, func)
            self.graph.add_edge(contract_id, func_id, edge_type="has_function")

        # Add modifiers
        for modifier in contract.modifiers:
            mod_id = f"modifier_{contract.name}_{modifier.name}"
            self.graph.add_node(
                mod_id,
                node_type="Modifier",
                name=modifier.name,
                visibility=str(modifier.visibility)
            )
            self.graph.add_edge(contract_id, mod_id, edge_type="has_modifier")

        # Add events
        for event in contract.events:
            event_id = f"event_{contract.name}_{event.name}"
            self.graph.add_node(
                event_id,
                node_type="Event",
                name=event.name
            )
            self.graph.add_edge(contract_id, event_id, edge_type="has_event")

    def _add_function_to_ast(self, contract: Contract, func: Function) -> str:
        """Add function and its components to AST"""
        func_id = f"func_{contract.name}_{func.name}_{func.signature_str}"

        self.graph.add_node(
            func_id,
            node_type="Function",
            name=func.name,
            visibility=str(func.visibility),
            is_constructor=func.is_constructor,
            is_fallback=func.is_fallback,
            is_receive=func.is_receive,
            payable=func.payable,
            view=func.view,
            pure=func.pure,
            signature=func.signature_str
        )

        # Add parameters
        for param in func.parameters:
            param_id = f"param_{func_id}_{param.name}"
            self.graph.add_node(
                param_id,
                node_type="Parameter",
                name=param.name,
                param_type=str(param.type)
            )
            self.graph.add_edge(func_id, param_id, edge_type="has_param")

        # Add return parameters
        for ret_param in func.returns:
            ret_id = f"return_{func_id}_{ret_param.name}"
            self.graph.add_node(
                ret_id,
                node_type="ReturnParameter",
                name=ret_param.name,
                param_type=str(ret_param.type)
            )
            self.graph.add_edge(func_id, ret_id, edge_type="returns")

        # Add local variables
        for var in func.local_variables:
            var_id = f"local_var_{func_id}_{var.name}"
            self.graph.add_node(
                var_id,
                node_type="LocalVariable",
                name=var.name,
                var_type=str(var.type)
            )
            self.graph.add_edge(func_id, var_id, edge_type="has_local_var")

        # Add function calls
        for internal_call in func.internal_calls:
            if hasattr(internal_call, 'name'):
                call_id = f"call_{func_id}_{internal_call.name}"
                if not self.graph.has_node(call_id):
                    self.graph.add_node(
                        call_id,
                        node_type="InternalCall",
                        name=str(internal_call.name)
                    )
                self.graph.add_edge(func_id, call_id, edge_type="calls")

        return func_id

    def get_graph_stats(self) -> Dict[str, int]:
        """Get statistics about the generated AST"""
        if not self.graph:
            return {}

        stats = {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges()
        }

        # Count by node type
        for node, data in self.graph.nodes(data=True):
            node_type = data.get("node_type", "Unknown")
            key = f"{node_type.lower()}_count"
            stats[key] = stats.get(key, 0) + 1

        return stats
