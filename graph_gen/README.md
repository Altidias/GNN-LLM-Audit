# Smart Contract Graph Generation

A system for generating graph representations of Solidity smart contracts for GNN vulnerability detection.

## Overview

The system generates Abstract Syntax Trees (AST), Control Flow Graphs (CFG), and Data Flow Graphs (DFG) from Solidity contracts. All graphs are stored in a unified format alongside the original source code.

The toolkit can be imported and used programmatically:

```python
from graph_gen import GraphGenerationOrchestrator, ContractData

orchestrator = GraphGenerationOrchestrator()
contract_data = orchestrator.generate_all_graphs("contract.sol")

ast_graph = contract_data.get_ast()
cfg_graph = contract_data.get_cfg()
dfg_graph = contract_data.get_dfg()
```

Loading previously generated data:

```python
from graph_gen import ContractData

contract_data = ContractData.load("output.pkl")
ast = contract_data.get_ast()
```

## Graph Representations

**Abstract Syntax Tree**: Captures hierarchical program structure including contracts, functions, state variables, parameters, modifiers, and events. Nodes represent program elements with attributes for visibility, mutability, and type information. Edges denote structural relationships such as inheritance, containment, and function calls.

**Control Flow Graph**: Models execution paths through function implementations. Nodes represent program points including entry points, expressions, conditionals, loops, and return statements. Edges indicate possible execution flow with annotations for conditional branches (true/false paths) and loop control (break/continue).

**Data Flow Graph**: Tracks data dependencies and variable usage patterns. Nodes represent variables (state, local, parameter, temporary), constants, and operations (binary, unary, function calls). Edges denote data flow relationships including assignments, operation inputs/outputs, and function arguments/returns.

## Output

Pickle format provides efficient binary serialization of ContractData objects containing source code and all three graphs. This format enables fast loading and preserves NetworkX graph objects with full attribute information.

JSON format exports graphs in a human-readable node-link representation. This format is useful for inspection, debugging, and interoperability with other tools but is larger and slower to process than pickle format.

Both formats can be generated simultaneously. Individual contracts produce separate files while batch processing also generates a unified dataset file containing all processed contracts.
