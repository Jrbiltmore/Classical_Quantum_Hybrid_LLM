
# state_manager.py
# Manages quantum states for each voxel, including editing and visualization.

import numpy as np
from typing import Dict, Tuple

class StateManager:
    """Class responsible for managing and updating the quantum states of voxels."""

    def __init__(self):
        self.quantum_states = {}  # Stores the quantum states for each voxel

    def create_state(self, voxel_id: Tuple[int, int, int], state_vector: np.ndarray):
        """Creates a new quantum state for the given voxel ID."""
        self.quantum_states[voxel_id] = state_vector

    def edit_state(self, voxel_id: Tuple[int, int, int], new_state_vector: np.ndarray):
        """Edits the quantum state for the given voxel ID."""
        if voxel_id in self.quantum_states:
            self.quantum_states[voxel_id] = new_state_vector
        else:
            raise ValueError(f"Quantum state for voxel {voxel_id} does not exist.")

    def get_state(self, voxel_id: Tuple[int, int, int]) -> np.ndarray:
        """Returns the quantum state for the given voxel ID."""
        return self.quantum_states.get(voxel_id, None)

    def delete_state(self, voxel_id: Tuple[int, int, int]):
        """Deletes the quantum state associated with the given voxel ID."""
        if voxel_id in self.quantum_states:
            del self.quantum_states[voxel_id]
        else:
            raise ValueError(f"No quantum state found for voxel {voxel_id}.")

    def list_states(self) -> Dict[Tuple[int, int, int], np.ndarray]:
        """Returns a dictionary of all quantum states managed by this class."""
        return self.quantum_states
