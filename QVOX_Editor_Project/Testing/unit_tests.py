
# unit_tests.py
# Unit tests for the core functionality of voxel and quantum state editing.

import unittest
import numpy as np
from Core.voxel_editor import VoxelEditor
from Core.quantum_state_editor import QuantumStateEditor

class TestVoxelEditor(unittest.TestCase):

    def setUp(self):
        self.editor = VoxelEditor(grid_size=(10, 10, 10))
        self.quantum_state_editor = QuantumStateEditor()

    def test_create_voxel(self):
        position = (1, 1, 1)
        attributes = {"spin": 0.5}
        quantum_state = np.array([1.0 + 0j, 0.0 + 0j])
        self.editor.create_voxel(position, attributes, quantum_state)
        voxel = self.editor.get_voxel(position)
        self.assertIsNotNone(voxel)
        self.assertEqual(voxel.position, position)
        self.assertEqual(voxel.attributes["spin"], 0.5)
        np.testing.assert_array_equal(voxel.quantum_state, quantum_state)

    def test_edit_voxel(self):
        position = (2, 2, 2)
        attributes = {"entropy": 0.7}
        quantum_state = np.array([0.5 + 0.5j, 0.5 - 0.5j])
        self.editor.create_voxel(position, attributes, quantum_state)
        new_attributes = {"entropy": 0.9}
        new_quantum_state = np.array([0.0 + 0.0j, 1.0 + 0.0j])
        self.editor.edit_voxel(position, new_attributes, new_quantum_state)
        voxel = self.editor.get_voxel(position)
        self.assertEqual(voxel.attributes["entropy"], 0.9)
        np.testing.assert_array_equal(voxel.quantum_state, new_quantum_state)

    def test_delete_voxel(self):
        position = (3, 3, 3)
        attributes = {"entanglement": 1.0}
        quantum_state = np.array([0.0 + 0.0j, 1.0 + 0.0j])
        self.editor.create_voxel(position, attributes, quantum_state)
        self.editor.delete_voxel(position)
        voxel = self.editor.get_voxel(position)
        self.assertIsNone(voxel)

if __name__ == '__main__':
    unittest.main()
