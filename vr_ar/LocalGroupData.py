# Let's extend the planetary and solar system rendering to include the Local Group of galaxies.

# We will create new data structures for galaxies in the Local Group and extend the renderers to handle both planetary and galactic objects.

# First, we need to extend PlanetaryData to include the Local Group, and then update the renderer.

# Extending PlanetaryData.py to include Local Group galaxies data
# LocalGroupData.py

import math

class LocalGroupData:
    def __init__(self):
        self.local_group_galaxies = self._initialize_galaxy_data()

    def _initialize_galaxy_data(self):
        \"\"\" Initializes real-time data for each galaxy in the Local Group \"\"\"
        return {
            "Milky Way": {"distance_from_earth_ly": 0, "diameter_ly": 100_000},
            "Andromeda": {"distance_from_earth_ly": 2_537_000, "diameter_ly": 220_000},
            "Triangulum": {"distance_from_earth_ly": 3_000_000, "diameter_ly": 60_000},
            "Large Magellanic Cloud": {"distance_from_earth_ly": 163_000, "diameter_ly": 14_000},
            "Small Magellanic Cloud": {"distance_from_earth_ly": 200_000, "diameter_ly": 7_000},
            "IC 1613": {"distance_from_earth_ly": 2_300_000, "diameter_ly": 7_000},
            "Leo I": {"distance_from_earth_ly": 820_000, "diameter_ly": 2_000},
            "Leo II": {"distance_from_earth_ly": 700_000, "diameter_ly": 1_000},
            "Sagittarius Dwarf": {"distance_from_earth_ly": 70_000, "diameter_ly": 10_000},
        }

    def get_galaxy_data(self):
        return self.local_group_galaxies

    def get_galaxy_position(self, galaxy_name):
        \"\"\" Calculate the position of a galaxy in the local group \"\"\"
        galaxy = self.local_group_galaxies.get(galaxy_name)
        if galaxy:
            return galaxy["distance_from_earth_ly"]
        return None

# Example usage
if __name__ == "__main__":
    local_group_data = LocalGroupData()
    print(local_group_data.get_galaxy_data())
    print("Andromeda distance from Earth:", local_group_data.get_galaxy_position("Andromeda"))
