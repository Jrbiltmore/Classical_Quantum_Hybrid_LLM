# brandons_file.py

import random

class EasterEggManager:
    def __init__(self):
        self.easter_eggs = {
            "golden_gun": "Unlocks a powerful golden gun with infinite ammo.",
            "infinite_lives": "Grants the player infinite lives.",
            "god_mode": "Enables invincibility for the player.",
            "bonus_level": "Unlocks a secret bonus level."
        }
        self.hacks = {
            "super_speed": "Increases the player's speed exponentially.",
            "extra_health": "Grants the player additional health points.",
            "auto_win": "Automatically wins the current level."
        }

    def trigger_easter_egg(self, code):
        \"\"\" Trigger an Easter egg based on a secret code. \"\"\"
        if code in self.easter_eggs:
            print(f"Easter egg activated: {self.easter_eggs[code]}")
            # Placeholder for actual game logic to apply the Easter egg
            return self.easter_eggs[code]
        else:
            print("Invalid Easter egg code.")
            return None

    def activate_hack(self, code):
        \"\"\" Activate a hack based on a secret code. \"\"\"
        if code in self.hacks:
            print(f"Hack activated: {self.hacks[code]}")
            # Placeholder for actual game logic to apply the hack
            return self.hacks[code]
        else:
            print("Invalid hack code.")
            return None

    def list_easter_eggs(self):
        \"\"\" List all available Easter eggs. \"\"\"
        return self.easter_eggs

    def list_hacks(self):
        \"\"\" List all available hacks. \"\"\"
        return self.hacks

    def generate_random_easter_egg(self):
        \"\"\" Generate a random Easter egg code. \"\"\"
        code = random.choice(list(self.easter_eggs.keys()))
        print(f"Random Easter egg generated: {code}")
        return code

    def generate_random_hack(self):
        \"\"\" Generate a random hack code. \"\"\"
        code = random.choice(list(self.hacks.keys()))
        print(f"Random hack generated: {code}")
        return code

# Example usage
if __name__ == "__main__":
    manager = EasterEggManager()
    print("Available Easter eggs:", manager.list_easter_eggs())
    print("Available hacks:", manager.list_hacks())
    
    # Example of triggering an Easter egg and hack
    manager.trigger_easter_egg("golden_gun")
    manager.activate_hack("super_speed")
    
    # Example of generating a random Easter egg and hack
    manager.generate_random_easter_egg()
    manager.generate_random_hack()
