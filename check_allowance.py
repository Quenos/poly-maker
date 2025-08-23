#!/usr/bin/env python3
"""Check USDC balance and allowance for Polymarket"""

import os
from dotenv import load_dotenv
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware
from eth_account import Account
import json

# Load environment variables
load_dotenv()

# Contract addresses
USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
POLYMARKET_EXCHANGE = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"  # CTF Exchange address
POLYMARKET_NEG_RISK_EXCHANGE = "0xC5d563A36AE78145C45a50134d48A1215220f80a"  # Neg Risk CTF Exchange

# Minimal ERC20 ABI for balance and allowance
ERC20_ABI = [
    {
        "inputs": [{"name": "account", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "", "type": "uint256"}],
        "type": "function",
        "constant": True
    },
    {
        "inputs": [
            {"name": "owner", "type": "address"},
            {"name": "spender", "type": "address"}
        ],
        "name": "allowance",
        "outputs": [{"name": "", "type": "uint256"}],
        "type": "function",
        "constant": True
    },
    {
        "inputs": [
            {"name": "spender", "type": "address"},
            {"name": "amount", "type": "uint256"}
        ],
        "name": "approve",
        "outputs": [{"name": "", "type": "bool"}],
        "type": "function"
    }
]

def check_balances_and_allowances():
    """Check USDC balance and allowances"""
    
    # Get private key and derive address
    private_key = os.getenv("PK")
    if not private_key:
        print("❌ PK not found in .env")
        return
        
    if not private_key.startswith("0x"):
        private_key = "0x" + private_key
    
    account = Account.from_key(private_key)
    address = account.address
    
    print(f"Checking wallet: {address}\n")
    
    # Connect to Polygon
    web3 = Web3(Web3.HTTPProvider("https://polygon-rpc.com"))
    web3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
    
    if not web3.is_connected():
        print("❌ Failed to connect to Polygon RPC")
        return
    
    # Get USDC contract
    usdc_contract = web3.eth.contract(address=USDC_ADDRESS, abi=ERC20_ABI)
    
    # Check balance
    balance_wei = usdc_contract.functions.balanceOf(address).call()
    balance_usdc = balance_wei / 10**6  # USDC has 6 decimals
    print(f"USDC Balance: {balance_usdc} USDC")
    
    # Check allowances
    print("\nAllowances:")
    
    # CTF Exchange allowance
    allowance_ctf = usdc_contract.functions.allowance(address, POLYMARKET_EXCHANGE).call()
    allowance_ctf_usdc = allowance_ctf / 10**6
    print(f"  CTF Exchange: {allowance_ctf_usdc} USDC")
    
    # Neg Risk CTF Exchange allowance
    allowance_neg = usdc_contract.functions.allowance(address, POLYMARKET_NEG_RISK_EXCHANGE).call()
    allowance_neg_usdc = allowance_neg / 10**6
    print(f"  Neg Risk Exchange: {allowance_neg_usdc} USDC")
    
    print("\n" + "="*50 + "\n")
    
    # Check if allowances are sufficient
    if allowance_ctf_usdc == 0 and allowance_neg_usdc == 0:
        print("❌ No allowances set! You need to approve the Polymarket contracts.")
        print("\nWould you like to approve the contracts? (y/n)")
        
        response = input().strip().lower()
        if response == 'y':
            approve_contracts(web3, usdc_contract, account, address)
    elif allowance_ctf_usdc < balance_usdc or allowance_neg_usdc < balance_usdc:
        print("⚠️  Allowances are lower than your balance.")
        print(f"   You can only trade up to {min(allowance_ctf_usdc, allowance_neg_usdc)} USDC")
    else:
        print("✅ Allowances are sufficient for trading!")

def approve_contracts(web3, usdc_contract, account, address):
    """Approve Polymarket contracts to spend USDC"""
    
    # Max approval amount (common practice)
    MAX_INT = 2**256 - 1
    
    print("\nApproving contracts...")
    
    # Get current nonce
    nonce = web3.eth.get_transaction_count(address)
    
    # Approve CTF Exchange
    print("\n1. Approving CTF Exchange...")
    try:
        # Build transaction
        tx = usdc_contract.functions.approve(
            POLYMARKET_EXCHANGE, 
            MAX_INT
        ).build_transaction({
            'from': address,
            'nonce': nonce,
            'gas': 100000,
            'gasPrice': web3.eth.gas_price,
            'chainId': 137  # Polygon chain ID
        })
        
        # Sign and send
        signed_tx = account.sign_transaction(tx)
        tx_hash = web3.eth.send_raw_transaction(signed_tx.raw_transaction)
        print(f"   Transaction sent: 0x{tx_hash.hex()}")
        
        # Wait for confirmation
        receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
        if receipt.status == 1:
            print("   ✅ CTF Exchange approved!")
        else:
            print("   ❌ Transaction failed!")
            
        nonce += 1
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
    
    # Approve Neg Risk Exchange
    print("\n2. Approving Neg Risk Exchange...")
    try:
        tx = usdc_contract.functions.approve(
            POLYMARKET_NEG_RISK_EXCHANGE, 
            MAX_INT
        ).build_transaction({
            'from': address,
            'nonce': nonce,
            'gas': 100000,
            'gasPrice': web3.eth.gas_price,
            'chainId': 137
        })
        
        signed_tx = account.sign_transaction(tx)
        tx_hash = web3.eth.send_raw_transaction(signed_tx.raw_transaction)
        print(f"   Transaction sent: 0x{tx_hash.hex()}")
        
        receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
        if receipt.status == 1:
            print("   ✅ Neg Risk Exchange approved!")
        else:
            print("   ❌ Transaction failed!")
            
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
    
    print("\n✅ Approval process complete!")
    print("You should now be able to trade on Polymarket.")

if __name__ == "__main__":
    check_balances_and_allowances()
