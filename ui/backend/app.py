
from flask import Flask, jsonify, request
from quantum_model import QuantumPCA
from classical_model import ClassicalNN
import os

app = Flask(__name__)

# Endpoint to retrieve hybrid model metrics
@app.route('/metrics', methods=['GET'])
def get_metrics():
    metrics = {
        'quantum_power': '5.2 TFLOPS',
        'classical_performance': '89% Accuracy',
        'hybrid_efficiency': '76% Utilization'
    }
    return jsonify(metrics)

# Endpoint to run Quantum PCA
@app.route('/run_quantum_pca', methods=['POST'])
def run_quantum_pca():
    data = request.json.get('data', None)
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    quantum_pca = QuantumPCA(num_qubits=2)
    result = quantum_pca.run_pca(data)
    return jsonify({'result': result})

# Endpoint to upload data
@app.route('/upload_data', methods=['POST'])
def upload_data():
    file = request.files.get('file')
    if file:
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        return jsonify({'message': f'File {file.filename} uploaded successfully'}), 200
    return jsonify({'error': 'No file uploaded'}), 400

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
