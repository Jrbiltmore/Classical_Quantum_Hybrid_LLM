
# query_executor.py
# This file executes Quantum SQL queries by interacting with the quantum engine and data layer.

from sql_interpreter import SQLInterpreter

class QueryExecutor:
    def __init__(self, num_qubits):
        self.interpreter = SQLInterpreter(num_qubits)

    def execute(self, query):
        """Executes the given Quantum SQL query and returns the result."""
        return self.interpreter.execute_query(query)

# Example usage:
# executor = QueryExecutor(2)
# result = executor.execute("APPLY HADAMARD 0; MEASURE")
