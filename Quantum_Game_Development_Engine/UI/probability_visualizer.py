# probability_visualizer.py
# Quantum_Game_Development_Engine/UI/probability_visualizer.py

import tkinter as tk
from tkinter import Canvas
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ProbabilityVisualizer:
    def __init__(self, master):
        """
        Initializes the ProbabilityVisualizer with the provided master widget.
        :param master: The parent widget for this visualizer (usually a root window or frame)
        """
        self.master = master
        self.frame = tk.Frame(self.master, bg='white')
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Title label
        self.title_label = tk.Label(self.frame, text="Quantum Probability Visualizer", font=('Arial', 18), bg='white')
        self.title_label.pack(pady=10)

        # Canvas for matplotlib plots
        self.canvas_frame = tk.Frame(self.frame, bg='white')
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.update_plot([0]*10)  # Initialize with zeros

    def update_plot(self, probabilities):
        """
        Updates the probability plot with new data.
        :param probabilities: A list of probability values to plot
        """
        self.ax.clear()
        self.ax.bar(range(len(probabilities)), probabilities, color='blue')
        self.ax.set_xlabel('Quantum States')
        self.ax.set_ylabel('Probability')
        self.ax.set_title('Quantum Probability Distribution')

        self.canvas.draw()

# Example usage
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Probability Visualizer")

    # Create and pack the probability visualizer
    prob_visualizer = ProbabilityVisualizer(root)

    # Example probabilities
    example_probabilities = [0.1, 0.2, 0.15, 0.25, 0.1, 0.05, 0.1, 0.05]
    prob_visualizer.update_plot(example_probabilities)

    root.mainloop()
