# level_manager.py
import json

class LevelManager:
    def __init__(self, levels_file):
        self.levels_file = levels_file
        self.levels = self.load_levels()
        self.current_level = None

    def load_levels(self):
        """Load game levels from a file."""
        try:
            with open(self.levels_file, 'r') as file:
                levels = json.load(file)
            print(f"Levels loaded from {self.levels_file}")
            return levels
        except FileNotFoundError:
            print(f"No levels file found. Starting with an empty level set.")
            return {}
        except json.JSONDecodeError:
            print(f"Error decoding JSON from the levels file.")
            return {}

    def save_levels(self):
        """Save the current game levels to a file."""
        with open(self.levels_file, 'w') as file:
            json.dump(self.levels, file, indent=4)
        print(f"Levels saved to {self.levels_file}")

    def add_level(self, level_id, level_data):
        """Add or update a game level."""
        self.levels[level_id] = level_data
        self.save_levels()
        print(f"Added/Updated level: {level_id}")

    def remove_level(self, level_id):
        """Remove a game level."""
        if level_id in self.levels:
            del self.levels[level_id]
            self.save_levels()
            print(f"Removed level: {level_id}")
        else:
            print(f"Level {level_id} not found.")

    def get_level(self, level_id):
        """Get the data of a specific game level."""
        return self.levels.get(level_id, None)

    def set_current_level(self, level_id):
        """Set the current level."""
        if level_id in self.levels:
            self.current_level = level_id
            print(f"Current level set to: {level_id}")
        else:
            print(f"Level {level_id} not found.")

    def get_current_level(self):
        """Get the current level data."""
        if self.current_level:
            return self.levels.get(self.current_level, None)
        else:
            print("No current level set.")
            return None

    def list_levels(self):
        """List all game levels."""
        return self.levels

# Example usage
if __name__ == "__main__":
    level_manager = LevelManager('Quantum_Game_Development_Engine/Data/Levels/levels.json')

    # Example adding a new level
    level_data = {
        'name': 'Quantum Arena',
        'difficulty': 3,
        'settings': {
            'gravity': 9.8,
            'time_of_day': 'midnight'
        }
    }
    level_manager.add_level('level_1', level_data)

    # Example setting and getting the current level
    level_manager.set_current_level('level_1')
    current_level = level_manager.get_current_level()
    print("Current Level Data:", current_level)

    # Example listing all levels
    all_levels = level_manager.list_levels()
    print("All Levels:", all_levels)

    # Example removing a level
    level_manager.remove_level('level_1')
