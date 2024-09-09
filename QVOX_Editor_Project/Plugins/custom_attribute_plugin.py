
# custom_attribute_plugin.py
# Enables users to define and visualize custom multidimensional attributes not natively supported by the QVOX Editor.

from typing import Dict, Tuple

class CustomAttributePlugin:
    """Plugin that allows users to define and visualize custom attributes for voxels."""

    def __init__(self):
        self.custom_attributes = {}

    def add_custom_attribute(self, voxel_id: Tuple[int, int, int], attribute_name: str, value: float):
        """Adds a custom attribute to a specific voxel."""
        if voxel_id not in self.custom_attributes:
            self.custom_attributes[voxel_id] = {}
        self.custom_attributes[voxel_id][attribute_name] = value
        print(f"Custom attribute '{attribute_name}' added to voxel {voxel_id}.")

    def get_custom_attributes(self, voxel_id: Tuple[int, int, int]) -> Dict[str, float]:
        """Returns all custom attributes for a given voxel."""
        return self.custom_attributes.get(voxel_id, {})

    def visualize_custom_attributes(self, voxel_id: Tuple[int, int, int]):
        """Provides a visual representation of the custom attributes for a voxel."""
        attributes = self.get_custom_attributes(voxel_id)
        for attr_name, attr_value in attributes.items():
            print(f"Voxel {voxel_id}: {attr_name} = {attr_value}")
