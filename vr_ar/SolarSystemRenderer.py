# SolarSystemRenderer.py

from PlanetaryData import PlanetaryData

class SolarSystemRenderer:
    def __init__(self):
        self.planetary_data = PlanetaryData()

    def render(self, time_in_days):
        \"\"\" Render the current state of the solar system with planet positions \"\"\"
        planetary_positions = self._calculate_planetary_positions(time_in_days)
        for planet, position in planetary_positions.items():
            print(f"{planet} is at position {position}")

    def _calculate_planetary_positions(self, time_in_days):
        \"\"\" Calculate the positions of planets for rendering \"\"\"
        positions = {}
        for planet in self.planetary_data.get_planetary_data():
            positions[planet] = self.planetary_data.get_orbital_position(planet, time_in_days)
        return positions

# Example usage
if __name__ == "__main__":
    renderer = SolarSystemRenderer()
    renderer.render(100)