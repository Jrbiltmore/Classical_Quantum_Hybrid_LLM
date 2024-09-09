# game_save_manager.py
import json
import os
from datetime import datetime

class GameSaveManager:
    def __init__(self, save_directory):
        self.save_directory = save_directory
        self.ensure_directory_exists()
    
    def ensure_directory_exists(self):
        """Ensure the save directory exists."""
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)
            print(f"Created save directory at {self.save_directory}")

    def save_game_state(self, game_state, filename):
        """Save the game state to a file."""
        file_path = os.path.join(self.save_directory, filename)
        with open(file_path, 'w') as file:
            json.dump(game_state, file, indent=4)
        print(f"Game state saved to {file_path}")
    
    def load_game_state(self, filename):
        """Load the game state from a file."""
        file_path = os.path.join(self.save_directory, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No such file: '{file_path}'")
        with open(file_path, 'r') as file:
            game_state = json.load(file)
        print(f"Game state loaded from {file_path}")
        return game_state

    def save_quantum_state(self, quantum_state, filename):
        """Save the quantum state to a file."""
        file_path = os.path.join(self.save_directory, filename)
        with open(file_path, 'w') as file:
            json.dump(quantum_state, file, indent=4)
        print(f"Quantum state saved to {file_path}")

    def load_quantum_state(self, filename):
        """Load the quantum state from a file."""
        file_path = os.path.join(self.save_directory, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No such file: '{file_path}'")
        with open(file_path, 'r') as file:
            quantum_state = json.load(file)
        print(f"Quantum state loaded from {file_path}")
        return quantum_state

    def list_saves(self):
        """List all saved game and quantum state files."""
        files = os.listdir(self.save_directory)
        return [f for f in files if os.path.isfile(os.path.join(self.save_directory, f))]

    def create_save_filename(self, base_name):
        """Create a unique filename for a save based on the current date and time."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{base_name}_{timestamp}.json"

    def delete_save(self, filename):
        """Delete a saved game or quantum state file."""
        file_path = os.path.join(self.save_directory, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted file {file_path}")
        else:
            raise FileNotFoundError(f"No such file: '{file_path}'")

# Example usage
if __name__ == "__main__":
    save_manager = GameSaveManager('Quantum_Game_Development_Engine/Data/Saves')

    # Example game state and quantum state
    game_state = {
        'current_turn': 5,
        'game_over': False
    }

    quantum_state = {
        'qubits': [
            {'id': 'q1', 'state': 'superposition', 'position': [0, 0, 0]},
            {'id': 'q2', 'state': 'entangled', 'position': [1, 0, 0]}
        ],
        'entanglements': [
            {'pair': ['q1', 'q2'], 'strength': 0.8}
        ]
    }

    # Save game and quantum states
    save_manager.save_game_state(game_state, save_manager.create_save_filename('game_state'))
    save_manager.save_quantum_state(quantum_state, save_manager.create_save_filename('quantum_state'))

    # Load game and quantum states
    loaded_game_state = save_manager.load_game_state(save_manager.list_saves()[0])
    loaded_quantum_state = save_manager.load_quantum_state(save_manager.list_saves()[1])

    # List all saved files
    print("Saved files:", save_manager.list_saves())
