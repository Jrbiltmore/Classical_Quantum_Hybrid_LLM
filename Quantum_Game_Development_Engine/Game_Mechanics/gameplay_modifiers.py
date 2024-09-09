# gameplay_modifiers.py
import json

class GameplayModifiers:
    def __init__(self, modifiers_file):
        self.modifiers_file = modifiers_file
        self.modifiers = self.load_modifiers()

    def load_modifiers(self):
        """Load gameplay modifiers from a file."""
        try:
            with open(self.modifiers_file, 'r') as file:
                modifiers = json.load(file)
            print(f"Modifiers loaded from {self.modifiers_file}")
            return modifiers
        except FileNotFoundError:
            print(f"No modifiers file found. Starting with an empty modifier set.")
            return {}
        except json.JSONDecodeError:
            print(f"Error decoding JSON from the modifiers file.")
            return {}

    def save_modifiers(self):
        """Save the current gameplay modifiers to a file."""
        with open(self.modifiers_file, 'w') as file:
            json.dump(self.modifiers, file, indent=4)
        print(f"Modifiers saved to {self.modifiers_file}")

    def add_modifier(self, key, value):
        """Add or update a gameplay modifier."""
        self.modifiers[key] = value
        self.save_modifiers()
        print(f"Added/Updated modifier: {key} = {value}")

    def remove_modifier(self, key):
        """Remove a gameplay modifier."""
        if key in self.modifiers:
            del self.modifiers[key]
            self.save_modifiers()
            print(f"Removed modifier: {key}")
        else:
            print(f"Modifier {key} not found.")

    def get_modifier(self, key):
        """Get the value of a specific gameplay modifier."""
        return self.modifiers.get(key, None)

    def list_modifiers(self):
        """List all gameplay modifiers."""
        return self.modifiers

# Example usage
if __name__ == "__main__":
    modifiers = GameplayModifiers('Quantum_Game_Development_Engine/Data/Modifiers/gameplay_modifiers.json')

    # Example adding a new modifier
    modifiers.add_modifier('quantum_boost', {'effect': 'increase_speed', 'amount': 10})

    # Example getting a modifier
    boost = modifiers.get_modifier('quantum_boost')
    print("Quantum Boost Modifier:", boost)

    # Example listing all modifiers
    all_modifiers = modifiers.list_modifiers()
    print("All Modifiers:", all_modifiers)

    # Example removing a modifier
    modifiers.remove_modifier('quantum_boost')
