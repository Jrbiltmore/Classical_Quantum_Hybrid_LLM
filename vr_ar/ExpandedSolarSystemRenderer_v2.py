# ExpandedSolarSystemRenderer_v2.py

from ExpandedCelestialData_v2 import ExpandedCelestialDataV2

class ExpandedSolarSystemRendererV2:
    def __init__(self):
        self.celestial_data = ExpandedCelestialDataV2()

    def render(self, time_in_days):
        \"\"\" Render the current state of the solar system, Local Group galaxies, deep space objects, and exoplanets \"\"\"
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
    renderer = ExpandedSolarSystemRendererV2()
    renderer.render(100)
