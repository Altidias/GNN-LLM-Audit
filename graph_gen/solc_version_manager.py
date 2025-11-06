"""
Solidity Compiler Version Manager
Automatically detects pragma version from contracts and switches solc version
"""

import re
import subprocess
import logging
from typing import Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class SolcVersionManager:
    """Manages Solidity compiler version selection based on pragma directives"""

    @staticmethod
    def extract_pragma_version(contract_source: str) -> Optional[str]:
        """
        Extract Solidity version from pragma directive

        Args:
            contract_source: Solidity source code

        Returns:
            Version string or None if not found
        """
        # Match various pragma patterns
        patterns = [
            r'pragma\s+solidity\s+\^?(\d+\.\d+\.\d+)',  # ^0.8.0 or 0.8.0
            r'pragma\s+solidity\s+>=?\s*(\d+\.\d+\.\d+)',  # >=0.8.0
            r'pragma\s+solidity\s+\^?(\d+\.\d+)',  # ^0.8
            r'pragma\s+solidity\s+>=?\s*(\d+\.\d+)',  # >=0.8
        ]

        for pattern in patterns:
            match = re.search(pattern, contract_source, re.IGNORECASE)
            if match:
                version = match.group(1)
                logger.debug(f"Found pragma version: {version}")
                return version

        return None

    @staticmethod
    def extract_pragma_from_file(file_path: str) -> Optional[str]:
        """
        Extract Solidity version from a contract file

        Args:
            file_path: Path to Solidity file

        Returns:
            Version string or None
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Only read first few lines for efficiency
                content = ''.join([f.readline() for _ in range(20)])
                return SolcVersionManager.extract_pragma_version(content)
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return None

    @staticmethod
    def parse_version_constraint(version_str: str) -> Tuple[str, str]:
        """
        Parse version constraint from pragma

        Args:
            version_str: Version string like "^0.8.0" or ">=0.8.0 <0.9.0"

        Returns:
            Tuple of (operator, version)
        """
        # Handle caret (^) - compatible with version
        if '^' in version_str:
            version = version_str.replace('^', '').strip()
            return ('compatible', version)

        # Handle >= or >
        if '>=' in version_str or '>' in version_str:
            match = re.search(r'>=?\s*(\d+\.\d+(?:\.\d+)?)', version_str)
            if match:
                return ('gte', match.group(1))

        # Handle exact version
        match = re.search(r'(\d+\.\d+(?:\.\d+)?)', version_str)
        if match:
            return ('exact', match.group(1))

        return ('exact', version_str)

    @staticmethod
    def select_best_version(pragma_version: str) -> str:
        """
        Select the best installed solc version based on pragma

        Args:
            pragma_version: Version from pragma directive

        Returns:
            Best matching version string
        """
        operator, version = SolcVersionManager.parse_version_constraint(pragma_version)

        # Parse version components
        version_parts = version.split('.')
        major = int(version_parts[0]) if len(version_parts) > 0 else 0
        minor = int(version_parts[1]) if len(version_parts) > 1 else 0
        patch = int(version_parts[2]) if len(version_parts) > 2 else 0

        # For caret (^), use same major.minor with latest patch
        if operator == 'compatible':
            if minor == 0 and patch == 0:
                # ^0.0.x -> only 0.0.x compatible
                recommended = f"{major}.{minor}.{patch}"
            elif major == 0:
                # ^0.8.0 -> 0.8.x compatible
                recommended = f"{major}.{minor}.0"
            else:
                # ^1.0.0 -> 1.x.x compatible
                recommended = f"{major}.{minor}.{patch}"
        else:
            # For >= or exact, use the specified version
            if len(version_parts) == 2:
                recommended = f"{major}.{minor}.0"
            else:
                recommended = version

        logger.info(f"Recommended solc version for pragma '{pragma_version}': {recommended}")
        return recommended

    @staticmethod
    def get_installed_versions() -> list[str]:
        """
        Get list of installed solc versions

        Returns:
            List of version strings
        """
        try:
            result = subprocess.run(
                ['solc-select', 'versions'],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                # Parse output to extract installed versions
                lines = result.stdout.strip().split('\n')
                versions = []
                for line in lines:
                    # Extract version numbers (e.g., "0.8.0")
                    match = re.search(r'(\d+\.\d+\.\d+)', line)
                    if match:
                        versions.append(match.group(1))
                return versions
            else:
                logger.warning("Failed to get installed solc versions")
                return []

        except Exception as e:
            logger.error(f"Error getting installed versions: {e}")
            return []

    @staticmethod
    def is_version_installed(version: str) -> bool:
        """
        Check if a specific solc version is installed

        Args:
            version: Version string to check

        Returns:
            True if installed, False otherwise
        """
        installed = SolcVersionManager.get_installed_versions()
        return version in installed

    @staticmethod
    def install_version(version: str) -> bool:
        """
        Install a specific solc version using solc-select

        Args:
            version: Version string to install

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Installing solc version {version}...")
            result = subprocess.run(
                ['solc-select', 'install', version],
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode == 0:
                logger.info(f"Successfully installed solc {version}")
                return True
            else:
                logger.error(f"Failed to install solc {version}: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Error installing solc {version}: {e}")
            return False

    @staticmethod
    def use_version(version: str, auto_install: bool = True) -> bool:
        """
        Switch to a specific solc version

        Args:
            version: Version string to use
            auto_install: Automatically install if not present

        Returns:
            True if successful, False otherwise
        """
        # Check if version is installed
        if not SolcVersionManager.is_version_installed(version):
            if auto_install:
                logger.info(f"Solc {version} not installed, installing...")
                if not SolcVersionManager.install_version(version):
                    return False
            else:
                logger.error(f"Solc {version} not installed")
                return False

        # Switch to the version
        try:
            logger.info(f"Switching to solc {version}...")
            result = subprocess.run(
                ['solc-select', 'use', version],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                logger.info(f"Successfully switched to solc {version}")
                return True
            else:
                logger.error(f"Failed to switch to solc {version}: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Error switching to solc {version}: {e}")
            return False

    @staticmethod
    def get_current_version() -> Optional[str]:
        """
        Get the currently active solc version

        Returns:
            Version string or None
        """
        try:
            result = subprocess.run(
                ['solc', '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                # Parse version from output
                match = re.search(r'Version:\s*(\d+\.\d+\.\d+)', result.stdout)
                if match:
                    return match.group(1)

            return None

        except Exception as e:
            logger.error(f"Error getting current solc version: {e}")
            return None

    @staticmethod
    def auto_select_and_use_version(contract_path: str, auto_install: bool = True) -> bool:
        """
        Automatically detect and switch to appropriate solc version for a contract

        Args:
            contract_path: Path to Solidity contract
            auto_install: Automatically install if needed

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Reading pragma from: {contract_path}")

        # Extract pragma version
        pragma_version = SolcVersionManager.extract_pragma_from_file(contract_path)

        if not pragma_version:
            logger.warning(f"No pragma version found in {contract_path}, using current version")
            current = SolcVersionManager.get_current_version()
            if current:
                logger.info(f"Continuing with solc {current}")
            return True

        logger.info(f"Found pragma solidity: {pragma_version}")

        # Select best version
        target_version = SolcVersionManager.select_best_version(pragma_version)
        logger.info(f"Target solc version: {target_version}")

        # Check if already using correct version
        current_version = SolcVersionManager.get_current_version()
        logger.info(f"Current solc version: {current_version}")

        if current_version == target_version:
            logger.info(f"Already using correct solc version {target_version}")
            return True

        logger.info(f"Need to switch from {current_version} to {target_version}")

        # Switch to target version
        success = SolcVersionManager.use_version(target_version, auto_install=auto_install)

        if success:
            logger.info(f"Successfully switched to solc {target_version}")
        else:
            logger.error(f"Failed to switch to solc {target_version}")

        return success
