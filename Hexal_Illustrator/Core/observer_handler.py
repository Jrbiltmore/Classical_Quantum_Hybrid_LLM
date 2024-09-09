import numpy as np

class ObserverHandler:
    def __init__(self, grid_size=(100, 100), initial_position=(50, 50), initial_angle=0):
        self.grid_size = grid_size
        self.position = np.array(initial_position)
        self.angle = initial_angle
        self.velocity = np.array([0, 0])

    def adjust_observer_view(self, new_position, new_angle):
        """
        Adjusts the observer's view based on the new position and angle.
        - new_position: Tuple representing the new position (x, y).
        - new_angle: The new angle of the observer in degrees.
        """
        self.position = np.array(new_position)
        self.angle = new_angle

    def move_observer(self, velocity):
        """
        Moves the observer across the grid based on the given velocity.
        Velocity is a tuple representing movement in the x and y directions.
        - velocity: Tuple representing the velocity (dx, dy).
        """
        self.velocity = np.array(velocity)
        self.position += self.velocity
        self.position = np.clip(self.position, [0, 0], np.array(self.grid_size) - 1)

    def apply_observer_effects(self, grid):
        """
        Applies observer effects to the grid. The observer's angle and position alter the rendering,
        simulating a perspective shift or a quantum collapse based on observation.
        - grid: The hexal grid to apply observer effects to.
        """
        shifted_grid = np.roll(grid, self.position.astype(int), axis=(0, 1))
        rotated_grid = np.rot90(shifted_grid, k=int(self.angle // 90))
        return rotated_grid

    def simulate_relativity_effects(self, grid, speed):
        """
        Simulates relativistic effects such as time dilation or length contraction based on the observer's speed.
        - grid: The hexal grid affected by relativity.
        - speed: The velocity of the observer relative to the speed of light (as a fraction, where 1.0 is the speed of light).
        """
        if speed >= 1.0:
            raise ValueError("Speed must be less than the speed of light.")
        
        time_dilation_factor = np.sqrt(1 - speed ** 2)
        contracted_grid = np.clip(grid * time_dilation_factor, 0, np.max(grid))
        return contracted_grid

    def calculate_doppler_shift(self, frequency, observer_velocity, source_velocity):
        """
        Calculates the Doppler shift in observed frequencies based on the observer's velocity relative to the source.
        - frequency: The original frequency emitted by the source.
        - observer_velocity: The velocity of the observer relative to the source.
        - source_velocity: The velocity of the source.
        Returns the observed frequency after Doppler shift.
        """
        speed_of_light = 299792458  # m/s
        relative_velocity = observer_velocity - source_velocity
        doppler_factor = np.sqrt((1 + relative_velocity / speed_of_light) / (1 - relative_velocity / speed_of_light))
        return frequency * doppler_factor

    def adjust_based_on_environment(self, environmental_factors):
        """
        Adjusts the observer's perspective and movement based on external environmental factors such as gravity, temperature, or electromagnetic fields.
        - environmental_factors: A dictionary containing various environmental conditions affecting the observer.
        """
        gravity = environmental_factors.get("gravity", 9.8)
        temperature = environmental_factors.get("temperature", 300)
        electromagnetic_field = environmental_factors.get("electromagnetic_field", np.array([0, 0, 0]))

        # Simulate gravity effect on movement (slows down if high gravity)
        self.velocity *= (1 / gravity)

        # Simulate temperature effects on perception (faster movement in higher temperatures)
        temperature_factor = np.clip(temperature / 300, 0.5, 2.0)
        self.position += self.velocity * temperature_factor

        # Apply effects of electromagnetic fields on observer perception
        self.position += electromagnetic_field[:2]

    def observe_quantum_collapse(self, quantum_grid, coords):
        """
        Simulates a quantum collapse event when the observer "measures" a specific hexal in the quantum grid.
        - quantum_grid: The grid containing quantum state information.
        - coords: The coordinates of the hexal to observe, causing a quantum collapse.
        """
        collapsed_value = np.abs(quantum_grid[coords]) ** 2
        quantum_grid[coords] = collapsed_value + 0j  # Collapse to a definite state (real number)
        return quantum_grid

    def record_observer_history(self, position_history, angle_history):
        """
        Records the observer's position and angle history for future analysis or replay.
        - position_history: List of observer positions over time.
        - angle_history: List of observer angles over time.
        """
        position_history.append(self.position.copy())
        angle_history.append(self.angle)

    def load_observer_state(self, saved_position, saved_angle):
        """
        Loads a previously saved observer state (position and angle).
        - saved_position: The saved observer position.
        - saved_angle: The saved observer angle.
        """
        self.position = np.array(saved_position)
        self.angle = saved_angle
