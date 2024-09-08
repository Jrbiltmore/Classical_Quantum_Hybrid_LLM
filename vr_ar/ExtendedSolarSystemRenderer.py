# ExtendedSolarSystemRenderer.py

from PlanetaryData import PlanetaryData
from LocalGroupData import LocalGroupData

class ExtendedSolarSystemRenderer:
    def __init__(self):
        self.planetary_data = PlanetaryData()
        self.local_group_data = LocalGroupData()

    def render(self, time_in_days):
        \"\"\" Render the current state of the solar system and Local Group galaxies \"\"\"
        planetary_positions = self._calculate_planetary_positions(time_in_days)
        galaxy_distances = self._calculate_galaxy_distances()
        
        print("Rendering Solar System:")
        for planet, position in planetary_positions.items():
            print(f"{planet} is at position {position}")

        print("\\nRendering Local Group Galaxies:")
        for galaxy, distance in galaxy_distances.items():
            print(f"{galaxy} is {distance} light-years from Earth")

    def _calculate_planetary_positions(self, time_in_days):
        \"\"\" Calculate the positions of planets for rendering \"\"\"
        positions = {}
        for planet in self.planetary_data.get_planetary_data():
            positions[planet] = self.planetary_data.get_orbital_position(planet, time_in_days)
        return positions

    def _calculate_galaxy_distances(self):
        \"\"\" Calculate the distances of galaxies in the Local Group for rendering \"\"\"
        distances = {}
        for galaxy in self.local_group_data.get_galaxy_data():
            distances[galaxy] = self.local_group_data.get_galaxy_position(galaxy)
        return distances

# Example usage
if __name__ == "__main__":
    renderer = ExtendedSolarSystemRenderer()
    renderer.render(100)
