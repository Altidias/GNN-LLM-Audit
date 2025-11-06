"""
DFG (Data Flow Graph) Generator for Solidity Smart Contracts
Uses Slither to extract and generate data flow representations tracking variable dependencies
"""

import networkx as nx
import os
import logging
from typing import Dict, Any, Optional, Set, List
from slither import Slither
from slither.core.declarations import Contract, Function
from slither.core.variables.variable import Variable
from slither.core.cfg.node import Node
from slither.slithir.operations import (
    Operation, Assignment, Binary, Unary, Index, Member,
    HighLevelCall, LowLevelCall, InternalCall, Transfer, Send,
    SolidityCall, Return
)
from slither.slithir.variables import (
    ReferenceVariable, TemporaryVariable, TupleVariable,
    Constant, StateIRVariable, LocalIRVariable
)

logger = logging.getLogger(__name__)


class DFGGenerator:
    """Generates Data Flow Graph representation of smart contracts"""

    def __init__(self):
        self.graph = None
        self.var_counter = 0

    def generate(self, contract_path: str, contract_name: Optional[str] = None, auto_install_solc: bool = True) -> nx.DiGraph:
        """
        Generate DFG from a Solidity contract

        Args:
            contract_path: Path to the Solidity contract file
            contract_name: Specific contract name to analyze (optional)
            auto_install_solc: Automatically install and switch solc version (default: True)

        Returns:
            NetworkX DiGraph representing the DFG
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
        self._build_dfg(contract)

        return self.graph

    def _build_dfg(self, contract: Contract):
        """Build DFG from contract's data dependencies"""
        contract_id = f"contract_{contract.name}"
        self.graph.add_node(
            contract_id,
            node_type="ContractScope",
            name=contract.name,
            label=f"Contract: {contract.name}"
        )

        # Add state variables as nodes
        state_var_nodes = {}
        for state_var in contract.state_variables:
            var_id = self._add_variable_node(
                contract.name,
                "state",
                state_var.name,
                str(state_var.type),
                is_storage=True
            )
            state_var_nodes[state_var.name] = var_id
            self.graph.add_edge(contract_id, var_id, edge_type="declares")

        # Process each function to build data flow
        for func in contract.functions:
            if func.is_implemented:
                self._add_function_dfg(contract, func, state_var_nodes)

    def _add_function_dfg(
        self,
        contract: Contract,
        func: Function,
        state_var_nodes: Dict[str, str]
    ):
        """Add function's data flow graph"""
        func_id = f"func_scope_{contract.name}_{func.name}_{id(func)}"

        self.graph.add_node(
            func_id,
            node_type="FunctionScope",
            function_name=func.name,
            label=f"Function: {func.name}"
        )

        # Track variable definitions within this function
        var_defs: Dict[str, str] = {}

        # Add function parameters
        for param in func.parameters:
            param_id = self._add_variable_node(
                f"{contract.name}_{func.name}",
                "parameter",
                param.name,
                str(param.type)
            )
            var_defs[param.name] = param_id
            self.graph.add_edge(func_id, param_id, edge_type="parameter")

        # Add local variables
        for local_var in func.local_variables:
            var_id = self._add_variable_node(
                f"{contract.name}_{func.name}",
                "local",
                local_var.name,
                str(local_var.type)
            )
            var_defs[local_var.name] = var_id
            self.graph.add_edge(func_id, var_id, edge_type="declares")

        # Process each node's IR operations for data flow
        for node in func.nodes:
            self._process_node_data_flow(
                node,
                contract.name,
                func.name,
                var_defs,
                state_var_nodes
            )

        # Add edges for state variable reads/writes
        for state_var_read in func.state_variables_read:
            if state_var_read.name in state_var_nodes:
                self.graph.add_edge(
                    state_var_nodes[state_var_read.name],
                    func_id,
                    edge_type="read_by"
                )

        for state_var_written in func.state_variables_written:
            if state_var_written.name in state_var_nodes:
                self.graph.add_edge(
                    func_id,
                    state_var_nodes[state_var_written.name],
                    edge_type="writes_to"
                )

    def _process_node_data_flow(
        self,
        node: Node,
        contract_name: str,
        func_name: str,
        var_defs: Dict[str, str],
        state_var_nodes: Dict[str, str]
    ):
        """Process a CFG node's IR operations to extract data dependencies"""
        if not hasattr(node, 'irs') or not node.irs:
            return

        for ir in node.irs:
            self._process_ir_operation(
                ir,
                contract_name,
                func_name,
                var_defs,
                state_var_nodes,
                node.node_id
            )

    def _process_ir_operation(
        self,
        ir: Operation,
        contract_name: str,
        func_name: str,
        var_defs: Dict[str, str],
        state_var_nodes: Dict[str, str],
        node_id: int
    ):
        """Process individual IR operation to create data flow edges"""
        operation_id = f"op_{contract_name}_{func_name}_{node_id}_{self.var_counter}"
        self.var_counter += 1

        # Handle assignments
        if isinstance(ir, Assignment):
            lvalue = ir.lvalue
            rvalue = ir.rvalue

            lvalue_id = self._get_or_create_var_node(
                lvalue, contract_name, func_name, var_defs, state_var_nodes
            )
            rvalue_id = self._get_or_create_var_node(
                rvalue, contract_name, func_name, var_defs, state_var_nodes
            )

            if lvalue_id and rvalue_id:
                self.graph.add_edge(
                    rvalue_id,
                    lvalue_id,
                    edge_type="assignment",
                    operation="=",
                    node_id=node_id
                )

        # Handle binary operations
        elif isinstance(ir, Binary):
            lvalue_id = self._get_or_create_var_node(
                ir.lvalue, contract_name, func_name, var_defs, state_var_nodes
            )

            # Add operation node
            self.graph.add_node(
                operation_id,
                node_type="BinaryOperation",
                operation=str(ir.type),
                label=f"Op: {ir.type}"
            )

            # Add edges from operands to operation
            left_id = self._get_or_create_var_node(
                ir.variable_left, contract_name, func_name, var_defs, state_var_nodes
            )
            right_id = self._get_or_create_var_node(
                ir.variable_right, contract_name, func_name, var_defs, state_var_nodes
            )

            if left_id:
                self.graph.add_edge(left_id, operation_id, edge_type="operand")
            if right_id:
                self.graph.add_edge(right_id, operation_id, edge_type="operand")

            # Edge from operation to result
            if lvalue_id:
                self.graph.add_edge(operation_id, lvalue_id, edge_type="result")

        # Handle unary operations
        elif isinstance(ir, Unary):
            lvalue_id = self._get_or_create_var_node(
                ir.lvalue, contract_name, func_name, var_defs, state_var_nodes
            )

            self.graph.add_node(
                operation_id,
                node_type="UnaryOperation",
                operation=str(ir.type),
                label=f"Op: {ir.type}"
            )

            rvalue_id = self._get_or_create_var_node(
                ir.rvalue, contract_name, func_name, var_defs, state_var_nodes
            )

            if rvalue_id:
                self.graph.add_edge(rvalue_id, operation_id, edge_type="operand")
            if lvalue_id:
                self.graph.add_edge(operation_id, lvalue_id, edge_type="result")

        # Handle function calls
        elif isinstance(ir, (HighLevelCall, LowLevelCall, InternalCall)):
            self.graph.add_node(
                operation_id,
                node_type="FunctionCall",
                function=str(ir.function) if hasattr(ir, 'function') else "unknown",
                call_type=type(ir).__name__,
                label=f"Call: {ir.function if hasattr(ir, 'function') else 'unknown'}"
            )

            # Add arguments as inputs
            if hasattr(ir, 'arguments'):
                for arg in ir.arguments:
                    arg_id = self._get_or_create_var_node(
                        arg, contract_name, func_name, var_defs, state_var_nodes
                    )
                    if arg_id:
                        self.graph.add_edge(arg_id, operation_id, edge_type="argument")

            # Add return value as output
            if hasattr(ir, 'lvalue') and ir.lvalue:
                lvalue_id = self._get_or_create_var_node(
                    ir.lvalue, contract_name, func_name, var_defs, state_var_nodes
                )
                if lvalue_id:
                    self.graph.add_edge(operation_id, lvalue_id, edge_type="returns")

        # Handle return statements
        elif isinstance(ir, Return):
            if hasattr(ir, 'values') and ir.values:
                for ret_val in ir.values:
                    ret_id = self._get_or_create_var_node(
                        ret_val, contract_name, func_name, var_defs, state_var_nodes
                    )
                    if ret_id:
                        self.graph.add_node(
                            operation_id,
                            node_type="Return",
                            label="Return"
                        )
                        self.graph.add_edge(ret_id, operation_id, edge_type="return_value")

    def _get_or_create_var_node(
        self,
        var: Any,
        contract_name: str,
        func_name: str,
        var_defs: Dict[str, str],
        state_var_nodes: Dict[str, str]
    ) -> Optional[str]:
        """Get existing or create new variable node"""
        if var is None:
            return None

        # Handle constants
        if isinstance(var, Constant):
            var_id = f"const_{contract_name}_{func_name}_{self.var_counter}"
            self.var_counter += 1
            self.graph.add_node(
                var_id,
                node_type="Constant",
                value=str(var.value),
                var_type=str(var.type) if hasattr(var, 'type') else "unknown",
                label=f"Const: {var.value}"
            )
            return var_id

        # Get variable name
        var_name = str(var.name) if hasattr(var, 'name') else str(var)

        # Check if it's a state variable
        if var_name in state_var_nodes:
            return state_var_nodes[var_name]

        # Check if it's already defined in function scope
        if var_name in var_defs:
            return var_defs[var_name]

        # Create new temporary/reference variable
        var_type = str(var.type) if hasattr(var, 'type') else "unknown"
        var_id = self._add_variable_node(
            f"{contract_name}_{func_name}",
            "temporary",
            var_name,
            var_type
        )
        var_defs[var_name] = var_id

        return var_id

    def _add_variable_node(
        self,
        scope: str,
        var_kind: str,
        var_name: str,
        var_type: str,
        is_storage: bool = False
    ) -> str:
        """Add a variable node to the graph"""
        var_id = f"var_{scope}_{var_kind}_{var_name}_{self.var_counter}"
        self.var_counter += 1

        self.graph.add_node(
            var_id,
            node_type="Variable",
            var_kind=var_kind,
            name=var_name,
            var_type=var_type,
            is_storage=is_storage,
            label=f"{var_kind}: {var_name}"
        )

        return var_id

    def get_graph_stats(self) -> Dict[str, int]:
        """Get statistics about the generated DFG"""
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

        # Count edge types
        edge_types = {}
        for _, _, data in self.graph.edges(data=True):
            edge_type = data.get("edge_type", "unknown")
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1

        stats['edge_types'] = edge_types

        return stats
