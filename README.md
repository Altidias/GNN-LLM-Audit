# Smart Contract Graph Generation Toolkit

A Python-based system for generating graph representations of Solidity smart contracts to support vulnerability detection and automated auditing workflows. The toolkit produces three complementary graph types from contract source code using static analysis.

## Overview

The system generates Abstract Syntax Trees (AST), Control Flow Graphs (CFG), and Data Flow Graphs (DFG) from Solidity contracts. All graphs are stored in a unified format alongside the original source code, enabling efficient reuse without reprocessing. The toolkit automatically manages Solidity compiler versions by parsing pragma directives and switching to the appropriate compiler version as needed.

## Architecture

The system consists of five core components:

**Graph Generators**: Three specialized modules (ast_generator.py, cfg_generator.py, dfg_generator.py) that interface with Slither to extract different program representations. Each generator produces a NetworkX directed graph with nodes and edges carrying semantic annotations relevant to that representation type.

**Version Manager**: The solc_version_manager.py module handles automatic detection of required Solidity compiler versions from pragma directives. It manages installation and switching of compiler versions via solc-select, ensuring contracts compile with their intended compiler version.

**Data Container**: The contract_data.py module provides ContractData and ContractDataset classes for unified storage. ContractData encapsulates source code and all three graphs in a single object with serialization support via Python pickle. ContractDataset manages collections of contracts for batch processing scenarios.

**Orchestrator**: The main.py module coordinates the generation pipeline, managing the sequence of AST, CFG, and DFG generation while handling compiler version switching and file I/O operations.

**CLI Interface**: A command-line interface supporting both single contract and batch processing modes with options for output format and verbosity control.

## Installation

The toolkit requires Python 3.8+ and depends on slither-analyzer for static analysis capabilities. Install dependencies using pip:

```
pip install -r requirements.txt
```

The solc-select tool must be available for compiler version management. Initial setup requires installing at least one Solidity compiler version:

```
solc-select install 0.8.0
solc-select use 0.8.0
```

## Usage

Single contract processing generates all three graph types and stores them in a pickle file:

```
python graph_gen/main.py contract.sol -o output.pkl
```

Batch processing handles directories of contracts:

```
python graph_gen/main.py contracts/ --batch --output-dir results
```

Optional JSON export provides human-readable graph representations for inspection:

```
python graph_gen/main.py contract.sol --json --json-dir output_json
```

The verbose flag enables detailed logging including version detection and switching operations:

```
python graph_gen/main.py contract.sol -v
```

## Programmatic Interface

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

## Compiler Version Management

The system automatically handles Solidity version compatibility by parsing pragma directives in the format:

```
pragma solidity ^0.8.0;
pragma solidity >=0.6.0 <0.9.0;
```

When processing a contract, the toolkit extracts the pragma version, determines the appropriate compiler version, checks if that version is installed, installs it if necessary, and switches to it before analysis. This ensures contracts are analyzed with their intended compiler version, avoiding compatibility issues and enabling accurate analysis of contracts written for different Solidity versions.

## Output Formats

Pickle format provides efficient binary serialization of ContractData objects containing source code and all three graphs. This format enables fast loading and preserves NetworkX graph objects with full attribute information.

JSON format exports graphs in a human-readable node-link representation. This format is useful for inspection, debugging, and interoperability with other tools but is larger and slower to process than pickle format.

Both formats can be generated simultaneously. Individual contracts produce separate files while batch processing also generates a unified dataset file containing all processed contracts.

## System Requirements

The toolkit requires Windows, Linux, or macOS with Python 3.8 or higher. The Slither static analysis framework and its dependencies must be installed. The solc-select tool manages Solidity compiler versions and requires network access for downloading compilers. Sufficient disk space is needed for compiler installations and generated graph data.

## Dependencies

Core dependencies include slither-analyzer for static analysis, networkx for graph data structures, and solc-select for compiler management. Optional dependencies include matplotlib for visualization capabilities.

## Design Considerations

The system uses a directory-switching approach to handle Windows path compatibility issues with Slither. Graphs are generated sequentially (AST, CFG, DFG) with the AST generator handling compiler version switching for the entire pipeline. Version-compatible attribute checking accommodates different Slither API versions across releases. NetworkX directed graphs provide a standard format compatible with graph neural network frameworks and enable graph algorithms for analysis tasks.

## File Organization

The graph_gen directory contains all core modules. The test_contracts directory provides sample contracts for testing including examples with different Solidity versions. Output files use the .pkl extension for pickle format and .json for JSON exports.
