# object_data_saver.py
import json
import os

class ObjectDataSaver:
    def __init__(self, data_directory):
        self.data_directory = data_directory
        self.ensure_directory_exists()

    def ensure_directory_exists(self):
        """Ensure the data directory exists."""
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)
            print(f"Created data directory at {self.data_directory}")

    def save_object_data(self, object_data, filename):
        """Save object data to a file."""
        file_path = os.path.join(self.data_directory, filename)
        with open(file_path, 'w') as file:
            json.dump(object_data, file, indent=4)
        print(f"Object data saved to {file_path}")

    def load_object_data(self, filename):
        """Load object data from a file."""
        file_path = os.path.join(self.data_directory, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No such file: '{file_path}'")
        with open(file_path, 'r') as file:
            object_data = json.load(file)
        print(f"Object data loaded from {file_path}")
        return object_data

    def list_object_files(self):
        """List all object data files in the directory."""
        files = os.listdir(self.data_directory)
        return [f for f in files if os.path.isfile(os.path.join(self.data_directory, f))]

    def create_object_filename(self, base_name):
        """Create a unique filename for an object based on the current date and time."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{base_name}_{timestamp}.json"

    def delete_object_data(self, filename):
        """Delete an object data file."""
        file_path = os.path.join(self.data_directory, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted file {file_path}")
        else:
            raise FileNotFoundError(f"No such file: '{file_path}'")

# Example usage
if __name__ == "__main__":
    data_saver = ObjectDataSaver('Quantum_Game_Development_Engine/Data/ObjectData')

    # Example object data
    object_data = {
        'object_id': 'obj123',
        'type': 'quantum_cube',
        'position': [10, 20, 30],
        'quantum_state': 'superposition'
    }

    # Save object data
    data_saver.save_object_data(object_data, data_saver.create_object_filename('object_data'))

    # Load object data
    loaded_object_data = data_saver.load_object_data(data_saver.list_object_files()[0])

    # List all saved object data files
    print("Object data files:", data_saver.list_object_files())
