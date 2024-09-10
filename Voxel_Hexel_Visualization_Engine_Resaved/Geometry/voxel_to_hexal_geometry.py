# Voxel to Hexal Geometry Converter for Voxel-Hexel Visualization Engine
# Converts voxel grid coordinates to hexagonal grid coordinates

def convert_voxel_to_hexal(x, y, z):
    """
    Convert voxel grid coordinates (x, y, z) to hexagonal grid coordinates (q, r).
    :param x: x-coordinate of the voxel grid.
    :param y: y-coordinate of the voxel grid.
    :param z: z-coordinate of the voxel grid (used for layering in hexal space).
    :return: Tuple representing (q, r) axial coordinates in the hexagonal grid.
    """
    q = x - (z - (z & 1)) // 2
    r = z
    return q, r

def convert_voxel_list_to_hexal(voxel_list):
    """
    Convert a list of voxel grid coordinates to hexagonal grid coordinates.
    :param voxel_list: List of voxel (x, y, z) coordinate tuples.
    :return: List of hexal (q, r) coordinate tuples.
    """
    return [convert_voxel_to_hexal(x, y, z) for x, y, z in voxel_list]
