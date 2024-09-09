# player_controller.py
class Player:
    def __init__(self, player_id, name, position=(0, 0), health=100, inventory=None):
        if inventory is None:
            inventory = []
        self.player_id = player_id
        self.name = name
        self.position = position
        self.health = health
        self.inventory = inventory

    def move(self, new_position):
        """Move the player to a new position."""
        self.position = new_position
        print(f"{self.name} moved to position {self.position}")

    def take_damage(self, amount):
        """Reduce the player's health by the specified amount."""
        self.health -= amount
        self.health = max(self.health, 0)  # Ensure health doesn't go below 0
        print(f"{self.name} took {amount} damage, health is now {self.health}")
        if self.health <= 0:
            self.die()

    def heal(self, amount):
        """Increase the player's health by the specified amount."""
        self.health += amount
        print(f"{self.name} healed by {amount}, health is now {self.health}")

    def pick_up_item(self, item):
        """Add an item to the player's inventory."""
        self.inventory.append(item)
        print(f"{self.name} picked up {item}")

    def use_item(self, item):
        """Use an item from the player's inventory."""
        if item in self.inventory:
            self.inventory.remove(item)
            print(f"{self.name} used {item}")
        else:
            print(f"{item} not found in {self.name}'s inventory")

    def interact_with_object(self, game_object):
        """Interact with a game object."""
        print(f"{self.name} is interacting with {game_object.name}")
        # Assume interaction logic is handled in the GameObject class
        game_object.interact('use', self)

    def die(self):
        """Handle player death."""
        print(f"{self.name} has died.")

    def get_player_info(self):
        """Retrieve information about the player."""
        return {
            'player_id': self.player_id,
            'name': self.name,
            'position': self.position,
            'health': self.health,
            'inventory': self.inventory
        }

# Example usage
if __name__ == "__main__":
    # Create a player
    player = Player(
        player_id='player_001',
        name='Hero',
        position=(0, 0),
        health=100,
        inventory=['sword', 'potion']
    )

    # Move the player
    player.move((10, 20))

    # Take damage and heal
    player.take_damage(30)
    player.heal(20)

    # Pick up and use an item
    player.pick_up_item('shield')
    player.use_item('potion')

    # Interact with a game object (assuming GameObject is imported or defined elsewhere)
    # game_object = GameObject(obj_id='obj_002', name='Ancient Relic', position=(5, 5), properties={})
    # player.interact_with_object(game_object)

    # Retrieve player information
    player_info = player.get_player_info()
    print("Player Info:", player_info)
