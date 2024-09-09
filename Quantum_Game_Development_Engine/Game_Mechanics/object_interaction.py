# object_interaction.py
class GameObject:
    def __init__(self, obj_id, name, position, properties):
        self.obj_id = obj_id
        self.name = name
        self.position = position
        self.properties = properties
        self.is_active = True

    def move(self, new_position):
        """Move the object to a new position."""
        self.position = new_position
        print(f"{self.name} moved to position {self.position}")

    def update_properties(self, new_properties):
        """Update the object's properties."""
        self.properties.update(new_properties)
        print(f"{self.name} properties updated to {self.properties}")

    def interact(self, action, *args):
        """Perform an interaction with the object."""
        if not self.is_active:
            print(f"{self.name} is not active and cannot be interacted with.")
            return

        if action == 'pick_up':
            self.pick_up()
        elif action == 'use':
            self.use(*args)
        elif action == 'modify':
            self.modify(*args)
        else:
            print(f"Unknown action {action} for object {self.name}")

    def pick_up(self):
        """Pick up the object."""
        print(f"{self.name} has been picked up.")

    def use(self, *args):
        """Use the object with optional arguments."""
        print(f"{self.name} has been used with arguments {args}")

    def modify(self, *args):
        """Modify the object with optional arguments."""
        modifications = args[0] if args else {}
        self.update_properties(modifications)
        print(f"{self.name} has been modified with {modifications}")

    def deactivate(self):
        """Deactivate the object."""
        self.is_active = False
        print(f"{self.name} has been deactivated.")

    def activate(self):
        """Activate the object."""
        self.is_active = True
        print(f"{self.name} has been activated.")

    def get_object_info(self):
        """Retrieve information about the object."""
        return {
            'obj_id': self.obj_id,
            'name': self.name,
            'position': self.position,
            'properties': self.properties,
            'is_active': self.is_active
        }

# Example usage
if __name__ == "__main__":
    # Create a game object
    obj = GameObject(
        obj_id='obj_001',
        name='Ancient Tome',
        position=(5, 10),
        properties={'type': 'book', 'material': 'leather', 'condition': 'worn'}
    )

    # Move the object
    obj.move((10, 15))

    # Update properties
    obj.update_properties({'condition': 'new', 'value': 100})

    # Interact with the object
    obj.interact('pick_up')
    obj.interact('use', 'read')
    obj.interact('modify', {'value': 120})

    # Deactivate and activate the object
    obj.deactivate()
    obj.interact('pick_up')
    obj.activate()
    obj.interact('pick_up')

    # Retrieve object information
    obj_info = obj.get_object_info()
    print("Object Info:", obj_info)
