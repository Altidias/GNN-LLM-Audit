// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * Deliberately Vulnerable Bank Contract for Graph Visualization Demo
 * Contains multiple common vulnerabilities that are visible in graph representations
 */
contract VulnerableBank {
    mapping(address => uint256) public balances;
    address public owner;
    uint256 public totalDeposits;

    event Deposit(address indexed user, uint256 amount);
    event Withdrawal(address indexed user, uint256 amount);

    constructor() {
        owner = msg.sender;
    }

    // VULNERABILITY 1: Reentrancy in withdraw function
    // The external call happens before state update
    function withdraw(uint256 amount) public {
        require(balances[msg.sender] >= amount, "Insufficient balance");

        // VULNERABLE: External call before state update
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");

        // State updated AFTER external call - reentrancy risk!
        balances[msg.sender] -= amount;
        totalDeposits -= amount;

        emit Withdrawal(msg.sender, amount);
    }

    // Safe deposit function for comparison
    function deposit() public payable {
        require(msg.value > 0, "Must deposit something");

        balances[msg.sender] += msg.value;
        totalDeposits += msg.value;

        emit Deposit(msg.sender, msg.value);
    }

    // VULNERABILITY 2: Integer overflow in bonus calculation
    // No overflow checking on multiplication
    function calculateBonus(uint256 balance, uint256 multiplier) public pure returns (uint256) {
        // VULNERABLE: No overflow protection
        uint256 bonus = balance * multiplier;
        return bonus;
    }

    // VULNERABILITY 3: Unprotected state change
    // Anyone can call this and manipulate totalDeposits
    function updateTotalDeposits(uint256 newTotal) public {
        // VULNERABLE: No access control
        totalDeposits = newTotal;
    }

    // Safe function with proper access control for comparison
    function emergencyWithdraw() public {
        require(msg.sender == owner, "Only owner");

        uint256 balance = address(this).balance;
        (bool success, ) = owner.call{value: balance}("");
        require(success, "Transfer failed");
    }

    // Helper function showing data flow
    function getBalance(address user) public view returns (uint256) {
        return balances[user];
    }

    // Function showing conditional logic in CFG
    function canWithdraw(address user, uint256 amount) public view returns (bool) {
        if (balances[user] >= amount) {
            if (amount > 0) {
                return true;
            }
        }
        return false;
    }
}
