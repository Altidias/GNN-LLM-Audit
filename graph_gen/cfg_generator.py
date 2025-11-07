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

        # Post-process to identify vulnerability patterns
        self._identify_reentrancy_patterns()

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
        is_entry: bool = False,
        edge_type_override: Optional[str] = None
    ) -> str:
        """Recursively process CFG nodes"""
        node_id = f"cfg_node_{contract_name}_{func_name}_{node.node_id}"

        # If already visited, just add edge and return
        if node.node_id in visited:
            if parent_id and self.graph.has_node(node_id):
                if not is_entry:
                    # Determine edge type based on parent and current node
                    edge_type = edge_type_override or self._determine_edge_type(parent_id, node)
                    self.graph.add_edge(parent_id, node_id, edge_type=edge_type)
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

        # Detect external calls (critical for reentrancy)
        node_attrs["contains_external_call"] = self._detect_external_call(node)
        node_attrs["contains_state_write"] = self._detect_state_write(node)

        # Detect loop patterns (for DOS vulnerability detection)
        if node.type in [NodeType.IFLOOP, NodeType.STARTLOOP]:
            node_attrs["loop_pattern"] = self._detect_loop_pattern(node)

        self.graph.add_node(node_id, **node_attrs)

        # After adding node, check if it's part of a reentrancy pattern
        # This requires checking predecessors, so we do it after node is added
        if node_attrs["contains_state_write"]:
            # Will be updated after all nodes are processed
            node_attrs["potential_reentrancy"] = False

        # Add edge from parent
        if parent_id and not is_entry:
            edge_type = edge_type_override or self._determine_edge_type(parent_id, node)
            self.graph.add_edge(parent_id, node_id, edge_type=edge_type)
        elif is_entry:
            self.graph.add_edge(parent_id, node_id, edge_type="entry")

        # Process conditional branches with specific edge types
        if node.type == NodeType.IF:
            # Add true and false branches with typed edges
            true_branch = node.son_true
            false_branch = node.son_false

            if true_branch:
                self._process_cfg_node(
                    true_branch, node_id, contract_name, func_name, visited,
                    edge_type_override="conditional_true"
                )

            if false_branch:
                self._process_cfg_node(
                    false_branch, node_id, contract_name, func_name, visited,
                    edge_type_override="conditional_false"
                )

        # Process loop structures with specific edge types
        elif node.type == NodeType.IFLOOP:
            # Loop condition node - body, back edge, or exit
            for son in node.sons:
                # Determine edge type based on successor type
                if son.type in [NodeType.STARTLOOP, NodeType.IFLOOP] and son.node_id <= node.node_id:
                    # Back edge to loop header
                    edge_type = "loop_back"
                elif son.type == NodeType.ENDLOOP:
                    # Exit from loop
                    edge_type = "loop_exit"
                else:
                    # Loop body
                    edge_type = "loop_body"
                self._process_cfg_node(son, node_id, contract_name, func_name, visited, edge_type_override=edge_type)

        else:
            # Process all successor nodes with sequential edges
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

        # Count edge types (important for RGCN)
        edge_type_counts = {}
        for _, _, data in self.graph.edges(data=True):
            edge_type = data.get('edge_type', 'unknown')
            edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1

        stats['edge_types'] = edge_type_counts

        # Count vulnerability-relevant patterns
        stats['external_call_nodes'] = sum(1 for _, d in self.graph.nodes(data=True) if d.get('contains_external_call', False))
        stats['state_write_nodes'] = sum(1 for _, d in self.graph.nodes(data=True) if d.get('contains_state_write', False))
        stats['guard_nodes'] = sum(1 for _, d in self.graph.nodes(data=True) if d.get('is_guard', False))
        stats['potential_reentrancy_nodes'] = sum(1 for _, d in self.graph.nodes(data=True) if d.get('potential_reentrancy', False))

        return stats

    def _determine_edge_type(self, parent_id: str, current_node: Node) -> str:
        """Determine the edge type based on parent and current node context"""
        # Get parent node data
        parent_data = self.graph.nodes.get(parent_id, {})
        parent_type = parent_data.get("cfg_node_type", "")
        parent_has_external_call = parent_data.get("contains_external_call", False)

        # Critical: Edge AFTER external call (reentrancy detection)
        if parent_has_external_call:
            return "post_external_call"

        # Edge TO external call (also important for pattern detection)
        if self._detect_external_call(current_node):
            return "to_external_call"

        # Loop-related edges
        if "LOOP" in parent_type:
            return "loop_flow"

        # Default sequential flow
        return "sequential"

    def _detect_external_call(self, node: Node) -> bool:
        """Detect if node contains external calls (critical for reentrancy)"""
        if not node.expression:
            return False

        expr_str = str(node.expression).lower()

        # Low-level calls
        if any(call in expr_str for call in ['.call(', '.call{', '.delegatecall(', '.staticcall(']):
            return True

        # High-level transfer methods
        if any(call in expr_str for call in ['.transfer(', '.send(']):
            return True

        # Check Slither's internal call detection
        if hasattr(node, 'high_level_calls') and node.high_level_calls:
            return True

        if hasattr(node, 'low_level_calls') and node.low_level_calls:
            return True

        return False

    def _detect_state_write(self, node: Node) -> bool:
        """Detect if node writes to state variables"""
        if not hasattr(node, 'state_variables_written'):
            return False

        return len(node.state_variables_written) > 0

    def _detect_loop_pattern(self, node: Node) -> str:
        """Detect loop patterns for DOS vulnerability detection"""
        if not node.expression:
            return "unknown"

        expr_str = str(node.expression).lower()

        # Check for unbounded loops (array.length, mapping iteration)
        if any(pattern in expr_str for pattern in ['.length', 'array', '[]']):
            return "array_iteration"

        # Check for gas-intensive operations in loop
        if any(pattern in expr_str for pattern in ['sstore', 'call', 'transfer']):
            return "gas_intensive"

        # Numeric bounded loops
        if any(char.isdigit() for char in expr_str):
            return "bounded"

        return "standard"

    def _identify_reentrancy_patterns(self):
        """
        Post-process graph to identify potential reentrancy patterns
        Pattern: external call → state write (without proper checks between)
        """
        for node_id, node_data in self.graph.nodes(data=True):
            # Skip if not a state write node
            if not node_data.get('contains_state_write', False):
                continue

            # Check all predecessors (within 2 hops) for external calls
            potential_reentrancy = False

            # Check immediate predecessors
            for pred_id in self.graph.predecessors(node_id):
                pred_data = self.graph.nodes[pred_id]
                edge_data = self.graph.edges[pred_id, node_id]

                # Direct reentrancy: external call → state write
                if edge_data.get('edge_type') == 'post_external_call':
                    potential_reentrancy = True
                    break

                # Check 2-hop predecessors (external call → other → state write)
                if pred_data.get('contains_external_call', False):
                    # Check if there's a guard between them
                    has_guard_between = pred_data.get('is_guard', False)
                    if not has_guard_between:
                        potential_reentrancy = True
                        break

            # Update node attribute
            if potential_reentrancy:
                self.graph.nodes[node_id]['potential_reentrancy'] = True
