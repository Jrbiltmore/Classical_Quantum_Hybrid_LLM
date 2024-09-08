# ExpandedSolarSystemRenderer.py

from ExpandedCelestialData import ExpandedCelestialData

class ExpandedSolarSystemRenderer:
    def __init__(self):
        self.celestial_data = ExpandedCelestialData()

    def render(self, time_in_days):
        \"\"\" Render the current state of the solar system, Local Group galaxies, and deep space objects \"\"\"
        celestial_positions = self._calculate_celestial_positions(time_in_days)
        
        print("Rendering Celestial Objects:")
        for obj, position in celestial_positions.items():
            print(f"{obj} is at a distance of {position} light-years from Earth")

    def _calculate_celestial_positions(self, time_in_days):
        \"\"\" Calculate the distances of celestial objects for rendering \"\"\"
        positions = {}
        for obj in self.celestial_data.get_celestial_data():
            positions[obj] = self.celestial_data.get_object_position(obj)
        return positions

# Example usage
if __name__ == "__main__":
    renderer = ExpandedSolarSystemRenderer()
    renderer.render(100)
