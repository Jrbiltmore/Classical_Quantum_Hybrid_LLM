# non_player_character_controller.py
import random

class NonPlayerCharacter:
    def __init__(self, npc_id, name, position, behavior_pattern):
        self.npc_id = npc_id
        self.name = name
        self.position = position
        self.behavior_pattern = behavior_pattern
        self.state = 'idle'

    def update_position(self, new_position):
        """Update the NPC's position."""
        self.position = new_position
        print(f"{self.name} moved to position {self.position}")

    def perform_behavior(self):
        """Execute the NPC's behavior pattern."""
        if self.behavior_pattern == 'wander':
            self.state = 'wandering'
            print(f"{self.name} is wandering around.")
        elif self.behavior_pattern == 'attack':
            self.state = 'attacking'
            print(f"{self.name} is attacking!")
        elif self.behavior_pattern == 'interact':
            self.state = 'interacting'
            print(f"{self.name} is interacting with the environment.")
        else:
            self.state = 'idle'
            print(f"{self.name} is idle.")

    def interact_with_player(self, player):
        """Handle interaction with a player character."""
        if self.state == 'interacting':
            print(f"{self.name} interacts with player {player.name}.")
        else:
            print(f"{self.name} is not in an interactive state.")

    def update_behavior(self, new_behavior):
        """Update the NPC's behavior pattern."""
        self.behavior_pattern = new_behavior
        print(f"{self.name}'s behavior pattern updated to {self.behavior_pattern}")

    def get_npc_info(self):
        """Retrieve information about the NPC."""
        return {
            'npc_id': self.npc_id,
            'name': self.name,
            'position': self.position,
            'behavior_pattern': self.behavior_pattern,
            'state': self.state
        }

# Example usage
if __name__ == "__main__":
    # Create an NPC
    npc = NonPlayerCharacter(
        npc_id='npc_001',
        name='Gorath',
        position=(10, 20),
        behavior_pattern='wander'
    )

    # Update position
    npc.update_position((15, 25))

    # Perform behavior
    npc.perform_behavior()

    # Create a mock player object for interaction
    class Player:
        def __init__(self, name):
            self.name = name

    player = Player(name='Hero')

    # Interact with player
    npc.interact_with_player(player)

    # Update behavior
    npc.update_behavior('attack')
    npc.perform_behavior()

    # Retrieve NPC information
    npc_info = npc.get_npc_info()
    print("NPC Info:", npc_info)
