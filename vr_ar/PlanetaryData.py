# PlanetaryData.py

import math

class PlanetaryData:
    def __init__(self):
        self.planets = self._initialize_planetary_data()

    def _initialize_planetary_data(self):
        \"\"\" Initializes real-time data for each celestial body in the solar system \"\"\"
        return {
            "Sun": {"distance_from_earth_km": 149_597_870, "diameter_km": 1_391_016, "orbital_period_days": 0},
            "Mercury": {"distance_from_earth_km": 91_691_000, "diameter_km": 4_879, "orbital_period_days": 87.97},
            "Venus": {"distance_from_earth_km": 41_400_000, "diameter_km": 12_104, "orbital_period_days": 224.7},
            "Earth": {"distance_from_sun_km": 149_597_870, "diameter_km": 12_742, "orbital_period_days": 365.25},
            "Mars": {"distance_from_earth_km": 78_340_000, "diameter_km": 6_779, "orbital_period_days": 687},
            "Jupiter": {"distance_from_earth_km": 628_730_000, "diameter_km": 139_820, "orbital_period_days": 4331},
            "Saturn": {"distance_from_earth_km": 1_275_000_000, "diameter_km": 116_460, "orbital_period_days": 10_759},
            "Uranus": {"distance_from_earth_km": 2_724_000_000, "diameter_km": 50_724, "orbital_period_days": 30_687},
            "Neptune": {"distance_from_earth_km": 4_351_000_000, "diameter_km": 49_244, "orbital_period_days": 60_190},
            "Pluto": {"distance_from_earth_km": 5_906_000_000, "diameter_km": 2_377, "orbital_period_days": 90_560},
            "Moon": {"distance_from_earth_km": 384_400, "diameter_km": 3_474, "orbital_period_days": 27.32}
        }

    def get_planetary_data(self):
        return self.planets

    def get_orbital_position(self, planet_name, time_in_days):
        \"\"\" Calculate the orbital position of a planet based on its orbital period \"\"\"
        planet = self.planets.get(planet_name)
        if planet and planet["orbital_period_days"] != 0:
            orbital_period = planet["orbital_period_days"]
            angle = (2 * math.pi * time_in_days) / orbital_period
            x = planet["distance_from_earth_km"] * math.cos(angle)
            y = planet["distance_from_earth_km"] * math.sin(angle)
            return x, y
        return 0, 0

# Example usage
if __name__ == "__main__":
    planetary_data = PlanetaryData()
    print(planetary_data.get_planetary_data())
    print("Mars position after 100 days:", planetary_data.get_orbital_position("Mars", 100))
