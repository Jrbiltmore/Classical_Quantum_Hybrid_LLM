# player_progress_saver.py
import json
import os
from datetime import datetime

class PlayerProgressSaver:
    def __init__(self, data_directory):
        self.data_directory = data_directory
        self.ensure_directory_exists()

    def ensure_directory_exists(self):
        """Ensure the data directory exists."""
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)
            print(f"Created data directory at {self.data_directory}")

    def save_player_progress(self, player_id, progress_data):
        """Save player progress to a file."""
        filename = self.create_progress_filename(player_id)
        file_path = os.path.join(self.data_directory, filename)
        with open(file_path, 'w') as file:
            json.dump(progress_data, file, indent=4)
        print(f"Player progress for {player_id} saved to {file_path}")

    def load_player_progress(self, player_id):
        """Load player progress from a file."""
        filename = self.create_progress_filename(player_id)
        file_path = os.path.join(self.data_directory, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No such file: '{file_path}'")
        with open(file_path, 'r') as file:
            progress_data = json.load(file)
        print(f"Player progress for {player_id} loaded from {file_path}")
        return progress_data

    def list_progress_files(self):
        """List all player progress files in the directory."""
        files = os.listdir(self.data_directory)
        return [f for f in files if os.path.isfile(os.path.join(self.data_directory, f))]

    def create_progress_filename(self, player_id):
        """Create a unique filename for player progress based on player ID and timestamp."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"progress_{player_id}_{timestamp}.json"

    def delete_player_progress(self, player_id):
        """Delete a player progress file."""
        filename = self.create_progress_filename(player_id)
        file_path = os.path.join(self.data_directory, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted player progress file {file_path}")
        else:
            raise FileNotFoundError(f"No such file: '{file_path}'")

# Example usage
if __name__ == "__main__":
    progress_saver = PlayerProgressSaver('Quantum_Game_Development_Engine/Data/PlayerProgress')

    # Example player progress data
    progress_data = {
        'player_id': 'player1',
        'level': 5,
        'experience': 12345,
        'inventory': ['quantum_sword', 'shield_of_eternity'],
        'quests_completed': ['quest_alpha', 'quest_beta']
    }

    # Save player progress
    progress_saver.save_player_progress('player1', progress_data)

    # Load player progress
    loaded_progress = progress_saver.load_player_progress('player1')

    # List all saved progress files
    print("Player progress files:", progress_saver.list_progress_files())

    # Delete player progress
    progress_saver.delete_player_progress('player1')
