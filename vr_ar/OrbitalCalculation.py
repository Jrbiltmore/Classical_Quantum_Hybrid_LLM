# OrbitalCalculation.py

import math

class OrbitalCalculation:
    def __init__(self, orbital_period_days, distance_from_earth_km):
        self.orbital_period_days = orbital_period_days
        self.distance_from_earth_km = distance_from_earth_km

    def calculate_position(self, time_in_days):
        \"\"\" Calculate the current position of a celestial body based on time \"\"\"
        angle = (2 * math.pi * time_in_days) / self.orbital_period_days
        x = self.distance_from_earth_km * math.cos(angle)
        y = self.distance_from_earth_km * math.sin(angle)
        return x, y

# Example usage
if __name__ == "__main__":
    mars_orbit = OrbitalCalculation(687, 78_340_000)
    position = mars_orbit.calculate_position(100)
    print(f"Mars position after 100 days: {position}")