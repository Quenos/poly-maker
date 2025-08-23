#!/usr/bin/env node
// Approve USDC from a Safe (Gnosis Safe) or EOA to Polymarket spender contracts on Polygon
// Uses existing helpers in this repo

const path = require('path');
require('dotenv').config({ path: path.resolve(__dirname, '..', '.env') });
const { ethers } = require('ethers');
const { safeAbi } = require('./safeAbi');
const { signAndExecuteSafeTransaction } = require('./safe-helpers');

const USDC_ADDRESS = '0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174';
const CTF_EXCHANGE = '0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E';
const NEG_RISK_EXCHANGE = '0xC5d563A36AE78145C45a50134d48A1215220f80a';
const NEG_RISK_ADAPTER = '0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296';
const CONDITIONAL_TOKENS = '0x4D97DCd97eC945f40cF65F87097ACe5EA0476045';

// Minimal ERC20 ABI
const ERC20_ABI = [
  'function allowance(address owner, address spender) view returns (uint256)',
  'function approve(address spender, uint256 amount) returns (bool)'
];

const ERC1155_ABI = [
  'function isApprovedForAll(address account, address operator) view returns (bool)',
  'function setApprovalForAll(address operator, bool approved)'
];

async function main() {
  const pk = process.env.PK;
  const funderAddress = process.env.BROWSER_ADDRESS; // funder can be EOA or Safe

  if (!pk || !funderAddress) {
    console.error('Missing PK or BROWSER_ADDRESS in .env');
    process.exit(1);
  }

  const rpc = process.env.POLYGON_RPC_URL || 'https://polygon-rpc.com';
  const provider = new ethers.providers.JsonRpcProvider(rpc);
  const signer = new ethers.Wallet(pk, provider);

  // Basic gas sanity check
  const matic = await provider.getBalance(signer.address);
  if (matic.eq(0)) {
    console.error('EOA has 0 MATIC. Send ~0.1 MATIC to', signer.address, 'to pay gas.');
    return; // Abort early; txs will fail without gas
  }

  // Gas settings (EIP-1559) — Polygon minimum prio fee ~25 gwei
  const latest = await provider.getBlock('latest');
  const baseFee = latest.baseFeePerGas || ethers.BigNumber.from('25000000000'); // 25 gwei fallback
  const priority = ethers.utils.parseUnits('35', 'gwei');
  const maxFee = baseFee.mul(12).div(10).add(priority); // 1.2x base + priority
  const gasOverrides = { maxFeePerGas: maxFee, maxPriorityFeePerGas: priority };

  const usdc = new ethers.Contract(USDC_ADDRESS, ERC20_ABI, provider);
  const ctf = new ethers.Contract(CONDITIONAL_TOKENS, ERC1155_ABI, provider);
  const funderCode = await provider.getCode(funderAddress);
  const isContract = funderCode && funderCode !== '0x';

  // Report current allowances
  const [allowCtf, allowNeg, allowAdapter, apprCtf, apprNeg, apprAdapter] = await Promise.all([
    usdc.allowance(funderAddress, CTF_EXCHANGE),
    usdc.allowance(funderAddress, NEG_RISK_EXCHANGE),
    usdc.allowance(funderAddress, NEG_RISK_ADAPTER),
    ctf.isApprovedForAll(funderAddress, CTF_EXCHANGE),
    ctf.isApprovedForAll(funderAddress, NEG_RISK_EXCHANGE),
    ctf.isApprovedForAll(funderAddress, NEG_RISK_ADAPTER),
  ]);

  console.log(isContract ? 'Funder (Safe):' : 'Funder (EOA):', funderAddress);
  console.log('EOA signer:', signer.address);
  console.log('Allowance CTF:', allowCtf.toString());
  console.log('Allowance NegRisk:', allowNeg.toString());
  console.log('Allowance NegRiskAdapter:', allowAdapter.toString());
  console.log('CTF isApprovedForAll (CTF_EXCHANGE):', apprCtf);
  console.log('CTF isApprovedForAll (NEG_RISK_EXCHANGE):', apprNeg);
  console.log('CTF isApprovedForAll (NEG_RISK_ADAPTER):', apprAdapter);

  const targets = [
    { name: 'CTF Exchange', spender: CTF_EXCHANGE },
    { name: 'Neg-Risk Exchange', spender: NEG_RISK_EXCHANGE },
    { name: 'Neg-Risk Adapter', spender: NEG_RISK_ADAPTER },
  ];

  const iface = new ethers.utils.Interface(ERC20_ABI);
  for (const t of targets) {
    const data = iface.encodeFunctionData('approve', [t.spender, ethers.constants.MaxUint256]);
    if (isContract) {
      // Safe path
      console.log(`Submitting Safe exec to approve USDC for ${t.name} (${t.spender})...`);
      try {
        const safe = new ethers.Contract(funderAddress, safeAbi, signer);
        const tx = await signAndExecuteSafeTransaction(signer, safe, USDC_ADDRESS, data, { gasLimit: 500000, ...gasOverrides });
        console.log('Sent tx:', tx.hash ? tx.hash : tx);
        const receipt = await tx.wait?.();
        if (receipt) {
          console.log('Receipt status:', receipt.status);
        }
      } catch (e) {
        console.error('Approval failed:', e.message || e);
      }
    } else {
      // EOA path
      console.log(`Sending approve tx from EOA for ${t.name} (${t.spender})...`);
      try {
        const usdcWithSigner = usdc.connect(signer);
        const tx = await usdcWithSigner.approve(t.spender, ethers.constants.MaxUint256, { gasLimit: 120000, ...gasOverrides });
        console.log('Sent tx:', tx.hash);
        const receipt = await tx.wait();
        console.log('Receipt status:', receipt.status);
      } catch (e) {
        console.error('Approval failed:', e.message || e);
      }
    }
  }

  // Ensure ERC1155 approvals for all operators (recommended by Polymarket tooling)
  const operators = [CTF_EXCHANGE, NEG_RISK_EXCHANGE, NEG_RISK_ADAPTER];
  for (const op of operators) {
    const already = await ctf.isApprovedForAll(funderAddress, op);
    if (!already) {
      console.log(`Setting setApprovalForAll on Conditional Tokens for operator ${op}...`);
      try {
        const tx = await ctf.connect(signer).setApprovalForAll(op, true, { gasLimit: 150000, ...gasOverrides });
        console.log('Sent tx:', tx.hash);
        const rc = await tx.wait();
        console.log('Receipt status:', rc.status);
      } catch (e) {
        console.error('setApprovalForAll failed:', e.message || e);
      }
    } else {
      console.log(`Already approved (CTF setApprovalForAll) for ${op}`);
    }
  }

  // Show final allowances
  const [allowCtf2, allowNeg2, allowAdapter2, apprCtf2, apprNeg2, apprAdapter2] = await Promise.all([
    usdc.allowance(funderAddress, CTF_EXCHANGE),
    usdc.allowance(funderAddress, NEG_RISK_EXCHANGE),
    usdc.allowance(funderAddress, NEG_RISK_ADAPTER),
    ctf.isApprovedForAll(funderAddress, CTF_EXCHANGE),
    ctf.isApprovedForAll(funderAddress, NEG_RISK_EXCHANGE),
    ctf.isApprovedForAll(funderAddress, NEG_RISK_ADAPTER),
  ]);
  console.log('Final Allowance CTF:', allowCtf2.toString());
  console.log('Final Allowance NegRisk:', allowNeg2.toString());
  console.log('Final Allowance NegRiskAdapter:', allowAdapter2.toString());
  console.log('CTF isApprovedForAll (CTF_EXCHANGE):', apprCtf2);
  console.log('CTF isApprovedForAll (NEG_RISK_EXCHANGE):', apprNeg2);
  console.log('CTF isApprovedForAll (NEG_RISK_ADAPTER):', apprAdapter2);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});


