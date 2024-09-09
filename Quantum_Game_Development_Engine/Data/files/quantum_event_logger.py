# quantum_event_logger.py
import json
import os
from datetime import datetime

class QuantumEventLogger:
    def __init__(self, log_directory):
        self.log_directory = log_directory
        self.ensure_directory_exists()

    def ensure_directory_exists(self):
        """Ensure the log directory exists."""
        if not os.path.exists(self.log_directory):
            os.makedirs(self.log_directory)
            print(f"Created log directory at {self.log_directory}")

    def log_event(self, event_type, quantum_event_type=None, intensity=None, triggered_by=None):
        """Log a quantum event."""
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'quantum_event_type': quantum_event_type,
            'intensity': intensity,
            'triggered_by': triggered_by
        }
        log_file = self.get_log_filename()
        file_path = os.path.join(self.log_directory, log_file)
        with open(file_path, 'a') as file:
            file.write(json.dumps(event) + '\n')
        print(f"Event logged: {event}")

    def get_log_filename(self):
        """Create a filename for the log file based on the current date."""
        return f"quantum_event_log_{datetime.now().strftime('%Y%m%d')}.log"

    def retrieve_logs(self):
        """Retrieve all logs from the log directory."""
        log_file = self.get_log_filename()
        file_path = os.path.join(self.log_directory, log_file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No such log file: '{file_path}'")
        logs = []
        with open(file_path, 'r') as file:
            for line in file:
                logs.append(json.loads(line.strip()))
        return logs

    def clear_logs(self):
        """Clear all logs in the log directory."""
        log_file = self.get_log_filename()
        file_path = os.path.join(self.log_directory, log_file)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Log file {file_path} cleared.")
        else:
            raise FileNotFoundError(f"No such log file: '{file_path}'")

# Example usage
if __name__ == "__main__":
    logger = QuantumEventLogger('Quantum_Game_Development_Engine/Data/Logs')

    # Example logging
    logger.log_event(
        event_type='quantum_event',
        quantum_event_type='superposition_shift',
        intensity=0.8,
        triggered_by='player1'
    )

    # Retrieve logs
    logs = logger.retrieve_logs()
    print("Retrieved logs:", logs)

    # Clear logs
    logger.clear_logs()
