# Attribute Manager for Voxel-Hexel Visualization Engine

class AttributeManager:
    """
    Manages multidimensional attributes such as quantum states, spin, entropy, and custom user-defined properties.
    Provides functionality to set, retrieve, and list attributes for specific grid positions.
    """
    
    def __init__(self):
        self.attributes = {}

    def set_attribute(self, position, attribute_name, value):
        """
        Set a multidimensional attribute for a given position in the voxel or hexal grid.
        :param position: Tuple representing the position in the grid (e.g., (x, y, z)).
        :param attribute_name: The name of the attribute (e.g., "spin", "entropy").
        :param value: The value of the attribute.
        """
        if position not in self.attributes:
            self.attributes[position] = {}
        self.attributes[position][attribute_name] = value

    def get_attribute(self, position, attribute_name):
        """
        Get the value of a specific attribute for a given position.
        :param position: Tuple representing the position in the grid.
        :param attribute_name: The name of the attribute to retrieve.
        :return: The value of the attribute, or None if it does not exist.
        """
        return self.attributes.get(position, {}).get(attribute_name, None)

    def list_attributes(self, position):
        """
        List all attributes associated with a specific position in the grid.
        :param position: Tuple representing the position in the grid.
        :return: A list of attribute names for the position.
        """
        return list(self.attributes.get(position, {}).keys())

    def remove_attribute(self, position, attribute_name):
        """
        Remove a specific attribute from a given position.
        :param position: Tuple representing the position in the grid.
        :param attribute_name: The name of the attribute to remove.
        """
        if position in self.attributes and attribute_name in self.attributes[position]:
            del self.attributes[position][attribute_name]
            if not self.attributes[position]:  # If no attributes left, remove the position entry
                del self.attributes[position]

    def clear_attributes(self, position=None):
        """
        Clear all attributes for a given position or for the entire grid.
        :param position: Optional. If provided, clears attributes only for that position. If None, clears all attributes.
        """
        if position:
            if position in self.attributes:
                del self.attributes[position]
        else:
            self.attributes.clear()
