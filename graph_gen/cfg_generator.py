"""
CFG (Control Flow Graph) Generator for Solidity Smart Contracts
Uses Slither to extract and generate control flow representations
"""

import networkx as nx
import os
import logging
from typing import Dict, Any, Optional, Set
from slither import Slither
from slither.core.declarations import Contract, Function
from slither.core.cfg.node import Node, NodeType

try:
    from .solc_version_manager import SolcVersionManager
except ImportError:
    from solc_version_manager import SolcVersionManager

logger = logging.getLogger(__name__)


class CFGGenerator:
    """Generates Control Flow Graph representation of smart contracts"""

    def __init__(self):
        self.graph = None

    def generate(self, contract_path: str, contract_name: Optional[str] = None, auto_install_solc: bool = True) -> nx.DiGraph:
        """
        Generate CFG from a Solidity contract

        Args:
            contract_path: Path to the Solidity contract file
            contract_name: Specific contract name to analyze (optional)
            auto_install_solc: Automatically install and switch solc version (default: True)

        Returns:
            NetworkX DiGraph representing the CFG
        """
        # Note: Version switching is handled by AST generator first in the pipeline
        # Windows path fix: change to contract directory and use relative path
        original_dir = os.getcwd()
        contract_path = os.path.abspath(contract_path)
        contract_dir = os.path.dirname(contract_path)
        contract_file = os.path.basename(contract_path)

        try:
            os.chdir(contract_dir)
            slither = Slither(contract_file)
        finally:
            os.chdir(original_dir)

        # Get the target contract
        if contract_name:
            contract = next((c for c in slither.contracts if c.name == contract_name), None)
            if not contract:
                raise ValueError(f"Contract '{contract_name}' not found in {contract_path}")
        else:
            contract = slither.contracts[0] if slither.contracts else None
            if not contract:
                raise ValueError(f"No contracts found in {contract_path}")

        self.graph = nx.DiGraph()
        self._build_cfg(contract)

        return self.graph

    def _build_cfg(self, contract: Contract):
        """Build CFG from contract functions"""
        # Add contract-level entry node
        contract_id = f"contract_{contract.name}"
        self.graph.add_node(
            contract_id,
            node_type="ContractEntry",
            name=contract.name,
            label=f"Contract: {contract.name}"
        )

        # Process each function in the contract
        for func in contract.functions:
            if func.is_implemented:  # Only process implemented functions
                func_entry = self._add_function_cfg(contract, func)
                # Link contract to function entry
                self.graph.add_edge(contract_id, func_entry, edge_type="contains")

    def _add_function_cfg(self, contract: Contract, func: Function) -> str:
        """Add function's control flow graph"""
        func_entry_id = f"func_entry_{contract.name}_{func.name}_{id(func)}"

        # Add function entry node
        self.graph.add_node(
            func_entry_id,
            node_type="FunctionEntry",
            function_name=func.name,
            visibility=str(func.visibility),
            is_constructor=func.is_constructor,
            payable=func.payable,
            view=func.view,
            pure=func.pure,
            label=f"Function: {func.name}"
        )

        # Track visited nodes to avoid cycles
        visited_nodes: Set[int] = set()

        # Process CFG nodes
        if func.entry_point:
            self._process_cfg_node(
                func.entry_point,
                func_entry_id,
                contract.name,
                func.name,
                visited_nodes,
                is_entry=True
            )

        return func_entry_id

    def _process_cfg_node(
        self,
        node: Node,
        parent_id: str,
        contract_name: str,
        func_name: str,
        visited: Set[int],
        is_entry: bool = False
    ) -> str:
        """Recursively process CFG nodes"""
        node_id = f"cfg_node_{contract_name}_{func_name}_{node.node_id}"

        # If already visited, just add edge and return
        if node.node_id in visited:
            if parent_id and self.graph.has_node(node_id):
                if not is_entry:
                    self.graph.add_edge(parent_id, node_id, edge_type="control_flow")
            return node_id

        visited.add(node.node_id)

        # Determine node type and label
        node_type_str = self._get_node_type_string(node.type)
        node_label = self._get_node_label(node)

        # Add node to graph with version-compatible attributes
        node_attrs = {
            "node_type": node_type_str,
            "cfg_node_type": str(node.type),
            "label": node_label,
            "expression": str(node.expression) if node.expression else "",
            "source_mapping": str(node.source_mapping) if node.source_mapping else "",
        }

        # Add optional attributes if they exist (version compatibility)
        if hasattr(node, 'contains_assembly') and callable(node.contains_assembly):
            node_attrs["contains_assembly"] = node.contains_assembly()
        if hasattr(node, 'can_reenter') and callable(node.can_reenter):
            node_attrs["can_reenter"] = node.can_reenter()
        if hasattr(node, 'can_send_eth') and callable(node.can_send_eth):
            node_attrs["can_send_eth"] = node.can_send_eth()

        # Detect guard conditions (require/assert/revert)
        expression_str = str(node.expression).lower() if node.expression else ""
        node_attrs["is_guard"] = False
        node_attrs["guard_type"] = None

        if "require(" in expression_str:
            node_attrs["is_guard"] = True
            node_attrs["guard_type"] = "require"
        elif "assert(" in expression_str:
            node_attrs["is_guard"] = True
            node_attrs["guard_type"] = "assert"
        elif "revert(" in expression_str:
            node_attrs["is_guard"] = True
            node_attrs["guard_type"] = "revert"

        self.graph.add_node(node_id, **node_attrs)

        # Add edge from parent
        if parent_id and not is_entry:
            self.graph.add_edge(parent_id, node_id, edge_type="control_flow")
        elif is_entry:
            self.graph.add_edge(parent_id, node_id, edge_type="entry")

        # Process conditional branches
        if node.type == NodeType.IF:
            # Add true and false branches
            true_branch = node.son_true
            false_branch = node.son_false

            if true_branch:
                true_id = self._process_cfg_node(
                    true_branch, node_id, contract_name, func_name, visited
                )
                # Update edge to indicate true branch
                if self.graph.has_edge(node_id, true_id):
                    self.graph[node_id][true_id]['branch'] = 'true'

            if false_branch:
                false_id = self._process_cfg_node(
                    false_branch, node_id, contract_name, func_name, visited
                )
                # Update edge to indicate false branch
                if self.graph.has_edge(node_id, false_id):
                    self.graph[node_id][false_id]['branch'] = 'false'

        else:
            # Process all successor nodes
            for son in node.sons:
                self._process_cfg_node(son, node_id, contract_name, func_name, visited)

        return node_id

    def _get_node_type_string(self, node_type: NodeType) -> str:
        """Convert NodeType enum to string"""
        type_map = {
            NodeType.ENTRYPOINT: "EntryPoint",
            NodeType.IF: "Conditional",
            NodeType.ENDIF: "EndIf",
            NodeType.EXPRESSION: "Expression",
            NodeType.RETURN: "Return",
            NodeType.THROW: "Throw",
            NodeType.VARIABLE: "Variable",
            NodeType.ASSEMBLY: "Assembly",
            NodeType.IFLOOP: "LoopCondition",
            NodeType.STARTLOOP: "LoopStart",
            NodeType.ENDLOOP: "LoopEnd",
            NodeType.BREAK: "Break",
            NodeType.CONTINUE: "Continue",
            NodeType.PLACEHOLDER: "Placeholder",
            NodeType.TRY: "Try",
            NodeType.CATCH: "Catch"
        }
        return type_map.get(node_type, str(node_type))

    def _get_node_label(self, node: Node) -> str:
        """Generate a readable label for the node"""
        if node.expression:
            expr_str = str(node.expression)
            # Truncate long expressions
            if len(expr_str) > 50:
                expr_str = expr_str[:47] + "..."
            return f"{self._get_node_type_string(node.type)}: {expr_str}"
        return self._get_node_type_string(node.type)

    def get_graph_stats(self) -> Dict[str, int]:
        """Get statistics about the generated CFG"""
        if not self.graph:
            return {}

        stats = {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "is_dag": nx.is_directed_acyclic_graph(self.graph)
        }

        # Count by node type
        for node, data in self.graph.nodes(data=True):
            node_type = data.get("node_type", "Unknown")
            key = f"{node_type.lower()}_count"
            stats[key] = stats.get(key, 0) + 1

        # Count branch edges
        branch_true = sum(1 for _, _, d in self.graph.edges(data=True) if d.get('branch') == 'true')
        branch_false = sum(1 for _, _, d in self.graph.edges(data=True) if d.get('branch') == 'false')

        stats['branch_true_count'] = branch_true
        stats['branch_false_count'] = branch_false

        return stats
