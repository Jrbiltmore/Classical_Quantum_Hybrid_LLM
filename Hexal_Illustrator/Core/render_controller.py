import matplotlib.pyplot as plt
import numpy as np

class RenderController:
    def __init__(self, hexal_grid):
        self.hexal_grid = hexal_grid
        self.rendered_frame = None

    def update_rendering_pipeline(self):
        """
        Updates the rendering pipeline by clearing previous frames and preparing for new rendering data.
        """
        self.clear_frame()

    def refresh_visualization(self):
        """
        Refreshes the visual display of the hexal grid, ensuring the latest state is rendered.
        Uses a 2D heatmap for visualizing the hexal grid.
        """
        self.render_frame()

    def clear_frame(self):
        """
        Clears the rendered frame, preparing the canvas for the next rendering step.
        """
        self.rendered_frame = np.zeros_like(self.hexal_grid)

    def render_frame(self):
        """
        Renders the current hexal grid using matplotlib to visualize the state of the grid.
        Applies color schemes and adds graphical elements based on quantum state data.
        """
        plt.imshow(self.hexal_grid, cmap='plasma')
        plt.colorbar()
        plt.show()

    def apply_shading_effects(self, mode="flat"):
        """
        Applies shading effects to the grid to improve depth perception or highlight specific areas.
        - mode: Defines the shading mode ('flat', 'gouraud', etc.).
        """
        if mode == "flat":
            self.render_frame()
        elif mode == "gouraud":
            # Advanced shading using gradients for smoother transitions
            plt.imshow(self.hexal_grid, cmap='plasma', interpolation='bicubic')
            plt.colorbar()
            plt.show()

    def adjust_lighting(self, light_intensity=1.0):
        """
        Adjusts the lighting intensity of the rendering, simulating light sources in the hexal environment.
        - light_intensity: A float value that adjusts the brightness of the rendering.
        """
        light_adjusted_grid = self.hexal_grid * light_intensity
        plt.imshow(light_adjusted_grid, cmap='plasma')
        plt.colorbar()
        plt.show()

    def add_annotations(self, annotations):
        """
        Adds annotations to the grid, useful for marking important features or quantum states.
        - annotations: A list of tuples (coords, label) where coords is the position and label is the text.
        """
        for (x, y), label in annotations:
            plt.text(x, y, label, fontsize=9, color='white', ha='center')
        self.render_frame()

    def export_render_to_image(self, filename):
        """
        Exports the current rendered frame to an image file (PNG, JPG, etc.).
        """
        plt.imsave(filename, self.hexal_grid, cmap='plasma')

    def render_quantum_states(self, quantum_grid):
        """
        Visualizes the quantum grid by representing real and imaginary parts, with custom color schemes.
        """
        real_part = np.real(quantum_grid)
        imag_part = np.imag(quantum_grid)
        
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(real_part, cmap='coolwarm')
        axs[0].set_title("Real Part of Quantum State")
        axs[1].imshow(imag_part, cmap='coolwarm')
        axs[1].set_title("Imaginary Part of Quantum State")
        plt.show()

    def apply_observer_effects(self, observer_position, observer_angle):
        """
        Applies observer effects such as perspective shifts or changes in the grid based on the observer's location and view angle.
        - observer_position: Tuple representing the observer's coordinates.
        - observer_angle: The angle at which the observer is viewing the grid.
        """
        shifted_grid = np.roll(self.hexal_grid, observer_position, axis=(0, 1))
        plt.imshow(shifted_grid, cmap='plasma')
        plt.title(f'Observer Angle: {observer_angle} degrees')
        plt.show()

    def visualize_grid_history(self, grid_history):
        """
        Visualizes the evolution of the hexal grid over time by rendering each state in the grid history.
        - grid_history: A list of previous grid states.
        """
        for i, grid in enumerate(grid_history):
            plt.imshow(grid, cmap='plasma')
            plt.title(f'Frame {i}')
            plt.pause(0.1)  # Simulate an animation by pausing between frames
        plt.show()

    def add_real_time_overlay(self, overlay_data):
        """
        Adds a real-time data overlay to the grid, such as temperature, pressure, or quantum attributes.
        - overlay_data: A 2D array representing the overlay data.
        """
        overlay_grid = self.hexal_grid + overlay_data
        plt.imshow(overlay_grid, cmap='plasma')
        plt.colorbar()
        plt.show()
