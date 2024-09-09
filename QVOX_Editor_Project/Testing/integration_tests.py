
# integration_tests.py
# Ensures that the UI, core engine, and file management work together correctly.

import unittest
from PyQt5.QtWidgets import QApplication
from Core.voxel_editor import VoxelEditor
from UI.main_window import MainWindow
from Data.file_loader import FileLoader
from Data.file_saver import FileSaver
import os
import numpy as np

class TestIntegration(unittest.TestCase):

    def setUp(self):
        self.app = QApplication([])  # PyQt5 application instance
        self.main_window = MainWindow()
        self.editor = VoxelEditor(grid_size=(10, 10, 10))
        self.file_loader = FileLoader()
        self.file_saver = FileSaver()

    def test_save_and_load_qvox_file(self):
        # Create and save a voxel
        position = (1, 1, 1)
        attributes = {"spin": 0.5}
        quantum_state = np.array([1.0 + 0j, 0.0 + 0j])
        self.editor.create_voxel(position, attributes, quantum_state)

        # Save to file
        test_filename = "test.qvox"
        self.file_saver.save_qvox_file(test_filename, self.editor.voxel_grid, self.editor.get_voxel(position).quantum_state)

        # Load from file
        loaded_data = self.file_loader.load_qvox_file(test_filename)
        loaded_grid = self.file_loader.load_voxel_grid(loaded_data)
        loaded_quantum_state = self.file_loader.load_quantum_states(loaded_data)

        # Verify the loaded voxel data
        self.assertEqual(loaded_grid[position], self.editor.get_voxel(position))
        np.testing.assert_array_equal(loaded_quantum_state[(1, 1, 1)], quantum_state)

        # Clean up
        os.remove(test_filename)

    def test_main_window_initialization(self):
        # Test if the main window initializes correctly
        self.assertIsNotNone(self.main_window)

if __name__ == '__main__':
    unittest.main()
