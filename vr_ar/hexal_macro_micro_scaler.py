# hexal_macro_micro_scaler.py

class HexalMacroMicroScaler:
    def __init__(self, scale_percent):
        self.scale_percent = scale_percent

    def scale(self, data):
        \"\"\"
        Dynamically scale data between macro and micro levels based on percentage.
        Supports zooming from 500% to -500%.
        \"\"\"
        return self._apply_scaling(data)

    def _apply_scaling(self, data):
        # Apply scaling based on the percentage input
        if self.scale_percent >= 100:
            return self._scale_zoom_in(data)
        elif self.scale_percent <= -100:
            return self._scale_zoom_out(data)
        else:
            return self._neutral_scale(data)

    def _scale_zoom_in(self, data):
        \"\"\" Zoom in, increase data scale \"\"\"
        scaling_factor = self.scale_percent / 100
        return [[value * scaling_factor for value in row] for row in data]

    def _scale_zoom_out(self, data):
        \"\"\" Zoom out, decrease data scale \"\"\"
        scaling_factor = abs(self.scale_percent) / 100
        return [[value / scaling_factor for value in row] for row in data]

    def _neutral_scale(self, data):
        \"\"\" Neutral scaling, no change \"\"\"
        return data

# Example usage
if __name__ == "__main__":
    scaler = HexalMacroMicroScaler(500)  # 500% zoom in
    data = [[1.0, 2.0], [3.0, 4.0]]
    scaled_data = scaler.scale(data)
    print(f"Scaled Data (Zoom In): {scaled_data}")

    scaler = HexalMacroMicroScaler(-500)  # -500% zoom out
    scaled_data = scaler.scale(data)
    print(f"Scaled Data (Zoom Out): {scaled_data}")