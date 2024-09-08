
# rest_api_endpoints.py
# REST API Endpoints for handling various quantum-classical operations

from flask import Flask, jsonify, request
from quantum_hybrid_integration.quantum_loss_functions import QuantumLossFunctions
from optimizations.hybrid_optimizer import HybridOptimizer

app = Flask(__name__)
quantum_loss_functions = QuantumLossFunctions()
optimizer = HybridOptimizer()

@app.route('/optimize', methods=['POST'])
def optimize_model():
    try:
        data = request.json
        classical_input = data.get('classical_data')
        quantum_input = data.get('quantum_data')

        # Process and optimize quantum-classical hybrid model
        optimized_classical = optimizer.optimize_classical(classical_input)
        optimized_quantum = optimizer.optimize_quantum(quantum_input)

        # Calculate quantum loss functions
        quantum_loss = quantum_loss_functions.compute_loss(optimized_quantum)

        return jsonify({
            'optimized_classical': optimized_classical,
            'optimized_quantum': optimized_quantum,
            'quantum_loss': quantum_loss
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
