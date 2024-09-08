# quantum_blockchain_ledger.py content placeholder
import hashlib
import json
from time import time
from typing import List, Dict

class QuantumBlockchainLedger:
    def __init__(self):
        self.chain: List[Dict] = []
        self.current_transactions: List[Dict] = []
        self.create_block(previous_hash='1', proof=100)  # Genesis block

    def create_block(self, proof: int, previous_hash: str = None) -> Dict:
        """
        Create a new block in the blockchain with quantum-resistant hashing.
        """
        block = {
            'index': len(self.chain) + 1,
            'timestamp': time(),
            'transactions': self.current_transactions,
            'proof': proof,
            'previous_hash': previous_hash or self.hash(self.chain[-1]),
        }
        self.current_transactions = []
        self.chain.append(block)
        return block

    def new_transaction(self, sender: str, recipient: str, amount: float) -> int:
        """
        Adds a new transaction to the list of transactions.
        """
        self.current_transactions.append({
            'sender': sender,
            'recipient': recipient,
            'amount': amount,
        })
        return self.last_block['index'] + 1

    def hash(self, block: Dict) -> str:
        """
        Quantum-resistant hashing using SHA-3 (Keccak256).
        """
        block_string = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha3_256(block_string).hexdigest()

    @property
    def last_block(self) -> Dict:
        return self.chain[-1]

    def proof_of_work(self, last_proof: int) -> int:
        """
        Simple Proof of Work algorithm:
        - Find a number p' such that hash(pp') contains leading 4 zeroes
        - p is the previous proof, and p' is the new proof
        """
        proof = 0
        while self.valid_proof(last_proof, proof) is False:
            proof += 1
        return proof

    def valid_proof(self, last_proof: int, proof: int) -> bool:
        """
        Validates the proof: Does hash(last_proof, proof) contain 4 leading zeroes?
        """
        guess = f'{last_proof}{proof}'.encode()
        guess_hash = hashlib.sha3_256(guess).hexdigest()
        return guess_hash[:4] == "0000"

    def is_chain_valid(self) -> bool:
        """
        Validate the blockchain by checking the consistency of the chain and hash values.
        """
        previous_block = self.chain[0]
        block_index = 1

        while block_index < len(self.chain):
            block = self.chain[block_index]
            # Check that the hash of the block is correct
            if block['previous_hash'] != self.hash(previous_block):
                return False
            # Check that the Proof of Work is correct
            if not self.valid_proof(previous_block['proof'], block['proof']):
                return False
            previous_block = block
            block_index += 1

        return True

if __name__ == '__main__':
    # Instantiate the Quantum Blockchain Ledger
    blockchain = QuantumBlockchainLedger()

    # Add a new transaction
    blockchain.new_transaction(sender="Alis", recipient="Jacob", amount=10)

    # Perform proof of work
    last_proof = blockchain.last_block['proof']
    proof = blockchain.proof_of_work(last_proof)

    # Add a new block to the chain
    previous_hash = blockchain.hash(blockchain.last_block)
    blockchain.create_block(proof, previous_hash)

    # Display the blockchain
    print("Blockchain:", blockchain.chain)

    # Verify the blockchain's validity
    print("Is blockchain valid?", blockchain.is_chain_valid())
