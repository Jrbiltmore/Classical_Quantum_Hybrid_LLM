import numpy as np

class HexalEngine:
    def __init__(self, grid_size=(100, 100), default_value=0, quantum_mode=False):
        self.grid_size = grid_size
        self.default_value = default_value
        self.grid = self.initialize_grid()
        self.quantum_mode = quantum_mode
        if self.quantum_mode:
            self.quantum_state_grid = self.initialize_quantum_grid()
        self.grid_history = []

    def initialize_grid(self):
        """
        Initializes a hexagonal grid with default values. Uses numpy for efficient matrix handling.
        """
        return np.full(self.grid_size, self.default_value, dtype=np.float32)

    def initialize_quantum_grid(self):
        """
        Initializes the quantum state grid, applying a random quantum superposition state for each hexal.
        Quantum states are complex numbers representing the wavefunction of each hexal.
        """
        return np.random.rand(self.grid_size[0], self.grid_size[1]) + 1j * np.random.rand(self.grid_size[0], self.grid_size[1])

    def update_grid(self, updates):
        """
        Updates the grid with new data.
        - updates: A dictionary where keys are tuples representing the hexal coordinates, and values are the new data.
        """
        for coords, value in updates.items():
            self.grid[coords] = value
        self.grid_history.append(self.grid.copy())

    def apply_quantum_updates(self, quantum_updates):
        """
        Updates the quantum state grid with new data, applying complex number updates.
        - quantum_updates: A dictionary where keys are tuples representing hexal coordinates, and values are complex quantum states.
        """
        for coords, quantum_state in quantum_updates.items():
            self.quantum_state_grid[coords] = quantum_state

    def calculate_neighborhood(self, x, y, radius=1):
        """
        Calculates and returns the values of hexals in the neighborhood around the given coordinates (x, y).
        Takes into account the hexagonal structure and returns the values of all adjacent hexals within a specified radius.
        """
        neighbors = []
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                if (i, j) != (0, 0):
                    neighbor_x = x + i
                    neighbor_y = y + j
                    if 0 <= neighbor_x < self.grid_size[0] and 0 <= neighbor_y < self.grid_size[1]:
                        neighbors.append(self.grid[neighbor_x, neighbor_y])
        return neighbors

    def run_simulation_step(self, rule):
        """
        Applies a user-defined rule to update the grid based on its current state. The rule is a function that takes in the current state and returns the new state.
        - rule: A function that defines how the grid should be updated at each step. Takes current state as input and returns updated state.
        """
        new_grid = np.copy(self.grid)
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                neighborhood = self.calculate_neighborhood(x, y)
                new_grid[x, y] = rule(self.grid[x, y], neighborhood)
        self.grid = new_grid
        self.grid_history.append(self.grid.copy())

    def simulate_quantum_collapse(self, observer_coords):
        """
        Simulates quantum state collapse based on observer interaction. If the observer observes a specific hexal, its quantum superposition collapses to a definite state.
        - observer_coords: Coordinates of the hexal observed by the observer.
        """
        collapsed_value = np.abs(self.quantum_state_grid[observer_coords]) ** 2  # Collapse to probability amplitude
        self.grid[observer_coords] = collapsed_value
        self.quantum_state_grid[observer_coords] = collapsed_value + 0j  # Collapse to a real number

    def revert_to_previous_state(self):
        """
        Reverts the grid to the previous state if a mistake is made or for simulation rollback purposes.
        """
        if len(self.grid_history) > 1:
            self.grid_history.pop()  # Remove the current state
            self.grid = self.grid_history[-1]  # Revert to the previous state

    def visualize_grid(self):
        """
        Outputs a simple visualization of the grid. This function can be extended to integrate with more complex rendering engines.
        """
        import matplotlib.pyplot as plt
        plt.imshow(self.grid, cmap="viridis")
        plt.colorbar()
        plt.show()

    def export_grid_to_file(self, filename):
        """
        Exports the current grid state to a file for further analysis or rendering.
        """
        np.savetxt(filename, self.grid, delimiter=',', fmt='%.4f')

    def import_grid_from_file(self, filename):
        """
        Imports a grid state from a file.
        """
        self.grid = np.loadtxt(filename, delimiter=',')

    def run_stochastic_simulation(self, rule, iterations=100):
        """
        Runs a stochastic simulation on the grid for a given number of iterations. At each iteration, the grid is updated based on the defined rule, with random fluctuations.
        - rule: The rule function that defines how the grid should evolve.
        - iterations: Number of iterations for the simulation.
        """
        for _ in range(iterations):
            random_updates = {
                (np.random.randint(self.grid_size[0]), np.random.randint(self.grid_size[1])): np.random.rand()
                for _ in range(int(self.grid_size[0] * self.grid_size[1] * 0.1))  # 10% random updates
            }
            self.update_grid(random_updates)
            self.run_simulation_step(rule)

    def apply_gradient_update(self, start_value, end_value):
        """
        Applies a gradient of values across the hexal grid, useful for visualizing transitions or simulating wave propagation.
        - start_value: The value to start the gradient with.
        - end_value: The value to end the gradient with.
        """
        gradient = np.linspace(start_value, end_value, num=self.grid_size[0])
        for x in range(self.grid_size[0]):
            self.grid[x, :] = gradient[x]
