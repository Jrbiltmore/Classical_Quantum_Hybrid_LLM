
# Voxel to Hexal Converter for Voxel-Hexel Visualization Engine

from .voxel_engine import VoxelGrid
from .hexal_engine import HexalGrid

def voxel_to_hexal(voxel_grid: VoxelGrid, hexal_grid: HexalGrid):
    # Convert voxel grid data into hexal grid data for visualization transitions
    for x in range(voxel_grid.dimensions[0]):
        for y in range(voxel_grid.dimensions[1]):
            for z in range(voxel_grid.dimensions[2]):
                value = voxel_grid.get_voxel(x, y, z)
                if value != 0:
                    q, r = convert_to_hexal_coordinates(x, y, z)
                    hexal_grid.set_hexal(q, r, value)

def convert_to_hexal_coordinates(x, y, z):
    # Conversion logic from voxel coordinates (x, y, z) to hexal coordinates (q, r)
    q = x - (z - (z & 1)) // 2
    r = z
    return q, r
