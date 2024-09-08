# perspective_handler.py

class PerspectiveHandler:
    def __init__(self, perspective="observer", time_scale="seconds"):
        self.perspective = perspective
        self.time_scale = time_scale

    def set_perspective(self, perspective):
        \"\"\" Set the current observation perspective \"\"\" 
        valid_perspectives = ["observer", "telescope", "microscope", "3rd_person"]
        if perspective in valid_perspectives:
            self.perspective = perspective
        else:
            raise ValueError(f"Invalid perspective: {perspective}. Choose from {valid_perspectives}.")

    def set_time_scale(self, time_scale):
        \"\"\" Set the current time scale \"\"\"
        valid_time_scales = ["atto", "femto", "pico", "nano", "micro", "milli", "seconds", "minutes", "hours", "days", "years", "light_years"]
        if time_scale in valid_time_scales:
            self.time_scale = time_scale
        else:
            raise ValueError(f"Invalid time scale: {time_scale}. Choose from {valid_time_scales}.")

    def get_perspective(self):
        \"\"\" Get the current observation perspective \"\"\"
        return self.perspective

    def get_time_scale(self):
        \"\"\" Get the current time scale \"\"\"
        return self.time_scale

    def apply_perspective(self, data):
        \"\"\" Adjust data visualization or rendering based on the current perspective and time scale \"\"\"
        data = self._apply_time_scale(data)
        if self.perspective == "observer":
            return self._observer_view(data)
        elif self.perspective == "telescope":
            return self._telescope_view(data)
        elif self.perspective == "microscope":
            return self._microscope_view(data)
        elif self.perspective == "3rd_person":
            return self._third_person_view(data)
        else:
            raise ValueError("Unknown perspective.")

    def _apply_time_scale(self, data):
        \"\"\" Adjust data based on time scale \"\"\"
        time_scales = {
            "atto": 10**(-18),
            "femto": 10**(-15),
            "pico": 10**(-12),
            "nano": 10**(-9),
            "micro": 10**(-6),
            "milli": 10**(-3),
            "seconds": 1,
            "minutes": 60,
            "hours": 3600,
            "days": 86400,
            "years": 31536000,
            "light_years": 9.461e+15  # meters per light-year
        }
        scale_factor = time_scales.get(self.time_scale, 1)
        return [[value * scale_factor for value in row] for row in data]

    def _observer_view(self, data):
        \"\"\" Apply observer view adjustments \"\"\"
        # Simulate normal human view range
        return data

    def _telescope_view(self, data):
        \"\"\" Apply telescope view adjustments \"\"\"
        # Zoom in for distant cosmic objects
        return [[value * 1000 for value in row] for row in data]

    def _microscope_view(self, data):
        \"\"\" Apply microscope view adjustments \"\"\"
        # Zoom in for atomic or subatomic scales
        return [[value * 1000000 for value in row] for row in data]

    def _third_person_view(self, data):
        \"\"\" Apply 3rd person cosmic to sub-Planck view adjustments \"\"\"
        # Scale between cosmic and sub-Planck levels
        return [[value * 10**(-35) for value in row] for row in data]

# Example usage
if __name__ == "__main__":
    handler = PerspectiveHandler("microscope", "nano")
    data = [[1.0, 2.0], [3.0, 4.0]]
    adjusted_data = handler.apply_perspective(data)
    print(f"Adjusted Data (Microscope, Nano Time Scale): {adjusted_data}")

    handler.set_perspective("telescope")
    handler.set_time_scale("light_years")
    adjusted_data = handler.apply_perspective(data)
    print(f"Adjusted Data (Telescope, Light Years Time Scale): {adjusted_data}")