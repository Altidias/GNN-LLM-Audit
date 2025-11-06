"""
Main Graph Generation Orchestrator
Coordinates AST, CFG, and DFG generation for smart contracts
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import logging

from ast_generator import ASTGenerator
from cfg_generator import CFGGenerator
from dfg_generator import DFGGenerator
from contract_data import ContractData, ContractDataset
from solc_version_manager import SolcVersionManager


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GraphGenerationOrchestrator:
    """
    Main orchestrator for generating all graph representations of smart contracts
    """

    def __init__(self):
        self.ast_generator = ASTGenerator()
        self.cfg_generator = CFGGenerator()
        self.dfg_generator = DFGGenerator()

    def generate_all_graphs(
        self,
        contract_path: str,
        contract_name: Optional[str] = None,
        save_path: Optional[str] = None,
        save_json: bool = False,
        json_dir: Optional[str] = None
    ) -> ContractData:
        """
        Generate all three graph representations for a contract

        Args:
            contract_path: Path to the Solidity contract file
            contract_name: Specific contract name to analyze (optional)
            save_path: Path to save the ContractData pickle file (optional)
            save_json: Whether to save graphs as JSON files
            json_dir: Directory to save JSON files (if save_json is True)

        Returns:
            ContractData object containing all graphs
        """
        logger.info(f"Processing contract: {contract_path}")
        logger.info("="*60)

        # Detect and display Solidity version from pragma
        pragma_version = SolcVersionManager.extract_pragma_from_file(contract_path)
        if pragma_version:
            logger.info(f"Detected pragma version: {pragma_version}")
            recommended_version = SolcVersionManager.select_best_version(pragma_version)
            logger.info(f"Recommended solc version: {recommended_version}")
        else:
            logger.warning("No pragma version found, using current solc version")

        # Check current solc version before processing
        current_version = SolcVersionManager.get_current_version()
        if current_version:
            logger.info(f"Current solc version: {current_version}")
        else:
            logger.warning("Could not determine current solc version")

        logger.info("="*60)

        # Read contract source
        with open(contract_path, 'r', encoding='utf-8') as f:
            contract_source = f.read()

        # Initialize ContractData
        contract_data = ContractData(
            contract_source=contract_source,
            contract_name=contract_name or Path(contract_path).stem,
            contract_path=contract_path
        )

        # Generate AST
        try:
            logger.info("Generating AST...")
            ast_graph = self.ast_generator.generate(contract_path, contract_name)
            contract_data.set_ast(ast_graph)
            ast_stats = self.ast_generator.get_graph_stats()
            logger.info(f"AST generated: {ast_stats['total_nodes']} nodes, {ast_stats['total_edges']} edges")
        except Exception as e:
            logger.error(f"Failed to generate AST: {e}")
            raise

        # Generate CFG
        try:
            logger.info("Generating CFG...")
            cfg_graph = self.cfg_generator.generate(contract_path, contract_name)
            contract_data.set_cfg(cfg_graph)
            cfg_stats = self.cfg_generator.get_graph_stats()
            logger.info(f"CFG generated: {cfg_stats['total_nodes']} nodes, {cfg_stats['total_edges']} edges")
        except Exception as e:
            logger.error(f"Failed to generate CFG: {e}")
            raise

        # Generate DFG
        try:
            logger.info("Generating DFG...")
            dfg_graph = self.dfg_generator.generate(contract_path, contract_name)
            contract_data.set_dfg(dfg_graph)
            dfg_stats = self.dfg_generator.get_graph_stats()
            logger.info(f"DFG generated: {dfg_stats['total_nodes']} nodes, {dfg_stats['total_edges']} edges")
        except Exception as e:
            logger.error(f"Failed to generate DFG: {e}")
            raise

        # Verify solc version after generation
        final_version = SolcVersionManager.get_current_version()
        if final_version:
            logger.info(f"Final solc version: {final_version}")

        # Save if requested
        if save_path:
            logger.info(f"Saving ContractData to {save_path}")
            contract_data.save(save_path)

        if save_json and json_dir:
            logger.info(f"Saving graphs as JSON to {json_dir}")
            contract_data.save_graphs_as_json(json_dir)

        logger.info("="*60)
        logger.info("Graph generation completed successfully")
        logger.info("="*60)
        return contract_data

    def generate_batch(
        self,
        contract_paths: list[str],
        output_dir: str,
        save_json: bool = False,
        save_dataset: bool = True
    ) -> ContractDataset:
        """
        Generate graphs for multiple contracts

        Args:
            contract_paths: List of paths to Solidity contract files
            output_dir: Directory to save outputs
            save_json: Whether to save individual JSONs
            save_dataset: Whether to save the entire dataset as one file

        Returns:
            ContractDataset containing all processed contracts
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        dataset = ContractDataset()

        for i, contract_path in enumerate(contract_paths, 1):
            logger.info(f"Processing contract {i}/{len(contract_paths)}: {contract_path}")

            try:
                # Generate name from filename
                contract_name = Path(contract_path).stem

                # Set up paths
                pickle_path = output_dir / f"{contract_name}.pkl"
                json_subdir = output_dir / "json" / contract_name if save_json else None

                # Generate graphs
                contract_data = self.generate_all_graphs(
                    contract_path=contract_path,
                    contract_name=contract_name,
                    save_path=str(pickle_path),
                    save_json=save_json,
                    json_dir=str(json_subdir) if json_subdir else None
                )

                dataset.add_contract(contract_data)

            except Exception as e:
                logger.error(f"Failed to process {contract_path}: {e}")
                continue

        # Save dataset
        if save_dataset:
            dataset_path = output_dir / "contract_dataset.pkl"
            logger.info(f"Saving complete dataset to {dataset_path}")
            dataset.save(str(dataset_path))

        logger.info(f"Batch processing complete. Processed {len(dataset)} contracts.")
        return dataset


def main():
    """Command-line interface for graph generation"""
    parser = argparse.ArgumentParser(
        description="Generate AST, CFG, and DFG graphs for Solidity smart contracts"
    )

    parser.add_argument(
        "contract_path",
        help="Path to Solidity contract file or directory"
    )

    parser.add_argument(
        "-n", "--name",
        help="Specific contract name to analyze (for files with multiple contracts)"
    )

    parser.add_argument(
        "-o", "--output",
        help="Output path for the pickle file (default: <contract_name>_graphs.pkl)"
    )

    parser.add_argument(
        "--output-dir",
        help="Output directory for batch processing"
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Save graphs as JSON files (in addition to pickle)"
    )

    parser.add_argument(
        "--json-dir",
        help="Directory to save JSON files (default: ./json_output)"
    )

    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all .sol files in the directory"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Initialize orchestrator
    orchestrator = GraphGenerationOrchestrator()

    contract_path = Path(args.contract_path)

    try:
        # Batch processing
        if args.batch or contract_path.is_dir():
            if not contract_path.is_dir():
                logger.error("Batch mode requires a directory path")
                sys.exit(1)

            # Find all .sol files
            sol_files = list(contract_path.glob("**/*.sol"))

            if not sol_files:
                logger.error(f"No .sol files found in {contract_path}")
                sys.exit(1)

            logger.info(f"Found {len(sol_files)} Solidity files")

            output_dir = args.output_dir or "./graph_outputs"

            dataset = orchestrator.generate_batch(
                contract_paths=[str(f) for f in sol_files],
                output_dir=output_dir,
                save_json=args.json
            )

            logger.info(f"Successfully processed {len(dataset)} contracts")

        # Single contract processing
        else:
            if not contract_path.exists():
                logger.error(f"Contract file not found: {contract_path}")
                sys.exit(1)

            output_path = args.output or f"{contract_path.stem}_graphs.pkl"
            json_dir = args.json_dir or "./json_output" if args.json else None

            contract_data = orchestrator.generate_all_graphs(
                contract_path=str(contract_path),
                contract_name=args.name,
                save_path=output_path,
                save_json=args.json,
                json_dir=json_dir
            )

            # Print summary
            print("\n" + "="*60)
            print("Graph Generation Summary")
            print("="*60)
            summary = contract_data.get_summary()
            for key, value in summary.items():
                print(f"{key}: {value}")
            print("="*60)

    except Exception as e:
        logger.error(f"Graph generation failed: {e}")
        if args.verbose:
            logger.exception("Detailed error:")
        sys.exit(1)


if __name__ == "__main__":
    main()
