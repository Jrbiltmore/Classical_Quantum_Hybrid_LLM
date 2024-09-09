
# performance_tests.py
# Performance benchmarks for rendering and real-time updates in large voxel grids.

import time
import unittest
import numpy as np
from Rendering.voxel_renderer import VoxelRenderer
from Rendering.real_time_update import RealTimeUpdate

class TestPerformance(unittest.TestCase):

    def setUp(self):
        self.voxel_renderer = VoxelRenderer()
        self.real_time_update = RealTimeUpdate()
        self.large_voxel_grid = np.ones((100, 100, 100))  # Simulating a large voxel grid

    def test_rendering_performance(self):
        # Test the performance of rendering a large voxel grid
        start_time = time.time()
        self.voxel_renderer.render_voxel_grid(self.large_voxel_grid)
        end_time = time.time()
        render_time = end_time - start_time
        print(f"Rendering large grid took {render_time:.2f} seconds")
        self.assertLess(render_time, 1.0)  # Expect rendering to be completed in under 1 second

    def test_real_time_update_performance(self):
        # Test the performance of real-time updates
        start_time = time.time()
        self.real_time_update.start_update_loop(lambda: None)
        time.sleep(0.1)  # Simulate 100ms real-time updates
        end_time = time.time()
        update_time = end_time - start_time
        print(f"Real-time update loop took {update_time:.2f} seconds")
        self.assertLess(update_time, 0.2)  # Expect the loop to run within the allotted time

if __name__ == '__main__':
    unittest.main()
