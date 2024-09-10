
# Hexal to Voxel Geometry Converter for Voxel-Hexel Visualization Engine
# Converts hexagonal grid coordinates to voxel grid coordinates

def convert_hexal_to_voxel(q, r, layer=0):
    """
    Convert hexagonal grid coordinates (q, r) to voxel grid coordinates (x, y, z).
    :param q: Axial q-coordinate of the hexal grid.
    :param r: Axial r-coordinate of the hexal grid.
    :param layer: Optional layer parameter to define height in voxel space (default is 0).
    :return: Tuple representing (x, y, z) coordinates in the voxel grid.
    """
    x = q + (r - (r & 1)) // 2
    y = r
    z = layer  # Use the layer as z-coordinate to maintain 3D structure
    return x, y, z

def convert_hexal_list_to_voxel(hexal_list, layer=0):
    """
    Convert a list of hexagonal grid coordinates to voxel grid coordinates.
    :param hexal_list: List of hexal (q, r) coordinate tuples.
    :param layer: Optional layer parameter to define height in voxel space.
    :return: List of voxel (x, y, z) coordinate tuples.
    """
    return [convert_hexal_to_voxel(q, r, layer) for q, r in hexal_list]
