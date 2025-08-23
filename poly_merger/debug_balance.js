#!/usr/bin/env node
// Diagnose Polygon balance visibility across multiple RPCs

const path = require('path');
require('dotenv').config({ path: path.resolve(__dirname, '..', '.env') });
const { ethers } = require('ethers');

const addr = process.env.BROWSER_ADDRESS;
if (!addr) {
  console.error('Missing BROWSER_ADDRESS in .env');
  process.exit(1);
}

const rpcs = [
  'https://polygon-rpc.com',
  'https://rpc.ankr.com/polygon',
  'https://polygon.llamarpc.com',
];

(async () => {
  console.log('Address:', addr);
  for (const url of rpcs) {
    try {
      const provider = new ethers.providers.JsonRpcProvider(url);
      const [network, bal] = await Promise.all([
        provider.getNetwork(),
        provider.getBalance(addr),
      ]);
      console.log(`\nRPC: ${url}`);
      console.log('ChainId:', network.chainId);
      console.log('Balance (wei):', bal.toString());
      console.log('Balance (MATIC):', ethers.utils.formatEther(bal));
    } catch (e) {
      console.log(`\nRPC: ${url}`);
      console.log('Error:', e.message || e);
    }
  }
})();


