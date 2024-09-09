# gameplay_status_panel.py
# Quantum_Game_Development_Engine/UI/gameplay_status_panel.py

import tkinter as tk

class GameplayStatusPanel:
    def __init__(self, master):
        """
        Initializes the GameplayStatusPanel with the provided master widget.
        :param master: The parent widget for this panel (usually a root window or frame)
        """
        self.master = master
        self.frame = tk.Frame(self.master, bg='lightblue')
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Title label
        self.title_label = tk.Label(self.frame, text="Gameplay Status", font=('Arial', 18), bg='lightblue')
        self.title_label.pack(pady=10)

        # Player Health
        self.health_label = tk.Label(self.frame, text="Player Health: 100", font=('Arial', 14), bg='lightblue')
        self.health_label.pack(pady=5)

        # Level Progress
        self.level_label = tk.Label(self.frame, text="Level: 1", font=('Arial', 14), bg='lightblue')
        self.level_label.pack(pady=5)

        # Score
        self.score_label = tk.Label(self.frame, text="Score: 0", font=('Arial', 14), bg='lightblue')
        self.score_label.pack(pady=5)

        # Other Metrics (e.g., time, ammo, etc.)
        self.other_metrics_label = tk.Label(self.frame, text="Other Metrics: N/A", font=('Arial', 14), bg='lightblue')
        self.other_metrics_label.pack(pady=5)

    def update_health(self, health):
        """
        Updates the player's health display.
        :param health: The new health value
        """
        self.health_label.config(text=f"Player Health: {health}")

    def update_level(self, level):
        """
        Updates the level display.
        :param level: The new level value
        """
        self.level_label.config(text=f"Level: {level}")

    def update_score(self, score):
        """
        Updates the score display.
        :param score: The new score value
        """
        self.score_label.config(text=f"Score: {score}")

    def update_other_metrics(self, metrics):
        """
        Updates other metrics display.
        :param metrics: A string representing other gameplay metrics
        """
        self.other_metrics_label.config(text=f"Other Metrics: {metrics}")

# Example usage
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Gameplay Status Panel")

    # Create and pack the gameplay status panel
    status_panel = GameplayStatusPanel(root)

    # Update the status panel with example data
    status_panel.update_health(85)
    status_panel.update_level(2)
    status_panel.update_score(1500)
    status_panel.update_other_metrics("Ammo: 30")

    root.mainloop()
