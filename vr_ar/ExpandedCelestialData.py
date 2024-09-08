# ExpandedCelestialData.py

import math

class ExpandedCelestialData:
    def __init__(self):
        self.celestial_objects = self._initialize_celestial_data()

    def _initialize_celestial_data(self):
        \"\"\" Initializes real-time data for galaxies, clusters, nebulae, and quasars \"\"\"
        return {
            # Local Group Galaxies
            "Milky Way": {"distance_from_earth_ly": 0, "diameter_ly": 100_000},
            "Andromeda": {"distance_from_earth_ly": 2_537_000, "diameter_ly": 220_000},
            "Triangulum": {"distance_from_earth_ly": 3_000_000, "diameter_ly": 60_000},

            # Galaxy Clusters
            "Virgo Cluster": {"distance_from_earth_ly": 54_000_000, "diameter_ly": 10_000_000},
            "Coma Cluster": {"distance_from_earth_ly": 320_000_000, "diameter_ly": 20_000_000},
            "Fornax Cluster": {"distance_from_earth_ly": 65_000_000, "diameter_ly": 8_000_000},

            # Nebulae
            "Crab Nebula": {"distance_from_earth_ly": 6_523, "diameter_ly": 11},
            "Orion Nebula": {"distance_from_earth_ly": 1_344, "diameter_ly": 24},
            "Eagle Nebula": {"distance_from_earth_ly": 7_000, "diameter_ly": 70},

            # Quasars
            "3C 273": {"distance_from_earth_ly": 2_443_000_000, "luminosity": "4 trillion Suns"},
            "ULAS J1120+0641": {"distance_from_earth_ly": 12_900_000_000, "luminosity": "60 trillion Suns"}
        }

    def get_celestial_data(self):
        return self.celestial_objects

    def get_object_position(self, object_name):
        \"\"\" Retrieve the distance of the object for rendering \"\"\"
        celestial_object = self.celestial_objects.get(object_name)
        if celestial_object:
            return celestial_object["distance_from_earth_ly"]
        return None

# Example usage
if __name__ == "__main__":
    celestial_data = ExpandedCelestialData()
    print(celestial_data.get_celestial_data())
    print("Virgo Cluster distance from Earth:", celestial_data.get_object_position("Virgo Cluster"))
