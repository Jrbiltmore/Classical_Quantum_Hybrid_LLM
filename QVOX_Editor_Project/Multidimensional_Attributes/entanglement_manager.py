
# entanglement_manager.py
# Handles quantum entanglement data, showing connections between voxels.

from typing import Dict, Tuple

class EntanglementManager:
    """Class responsible for managing and visualizing quantum entanglement between voxels."""

    def __init__(self):
        self.entanglement_map = {}  # Stores entanglement relationships between voxel pairs

    def create_entanglement(self, voxel1: Tuple[int, int, int], voxel2: Tuple[int, int, int], strength: float):
        """Creates an entanglement relationship between two voxels with a given strength."""
        if voxel1 not in self.entanglement_map:
            self.entanglement_map[voxel1] = {}
        self.entanglement_map[voxel1][voxel2] = strength

        if voxel2 not in self.entanglement_map:
            self.entanglement_map[voxel2] = {}
        self.entanglement_map[voxel2][voxel1] = strength

    def remove_entanglement(self, voxel1: Tuple[int, int, int], voxel2: Tuple[int, int, int]):
        """Removes the entanglement relationship between two voxels."""
        if voxel1 in self.entanglement_map and voxel2 in self.entanglement_map[voxel1]:
            del self.entanglement_map[voxel1][voxel2]
        if voxel2 in self.entanglement_map and voxel1 in self.entanglement_map[voxel2]:
            del self.entanglement_map[voxel2][voxel1]

    def get_entanglement_strength(self, voxel1: Tuple[int, int, int], voxel2: Tuple[int, int, int]) -> float:
        """Returns the strength of the entanglement between two voxels, or 0 if no entanglement exists."""
        return self.entanglement_map.get(voxel1, {}).get(voxel2, 0.0)

    def is_entangled(self, voxel1: Tuple[int, int, int], voxel2: Tuple[int, int, int]) -> bool:
        """Checks if two voxels are entangled."""
        return voxel2 in self.entanglement_map.get(voxel1, {})

    def list_entangled_voxels(self, voxel: Tuple[int, int, int]) -> Dict[Tuple[int, int, int], float]:
        """Returns a list of all voxels entangled with the specified voxel and their entanglement strengths."""
        return self.entanglement_map.get(voxel, {})

