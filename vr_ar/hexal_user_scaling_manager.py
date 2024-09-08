# hexal_user_scaling_manager.py

class HexalUserScalingManager:
    def __init__(self, user_input):
        self.user_input = user_input

    def manage_scaling(self):
        \"\"\" Manage scaling decisions based on user input, adjusting between 500% to -500% zoom levels. \"\"\"
        return self._determine_scale_percentage()

    def _determine_scale_percentage(self):
        \"\"\" Convert user input (e.g., zoom in/out) to a scale percentage. \"\"\"
        if self.user_input == "zoom_in":
            return 500
        elif self.user_input == "zoom_out":
            return -500
        else:
            return 100  # Default neutral scaling

# Example usage
if __name__ == "__main__":
    scaling_manager = HexalUserScalingManager("zoom_in")
    scale_percentage = scaling_manager.manage_scaling()
    print(f"Determined Scale Percentage: {scale_percentage}")

    scaling_manager = HexalUserScalingManager("zoom_out")
    scale_percentage = scaling_manager.manage_scaling()
    print(f"Determined Scale Percentage: {scale_percentage}")