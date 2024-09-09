
# QVOX Editor API Reference

This document provides a reference for the public API of the **QVOX Editor**, covering core functionality for manipulating voxels, quantum states, and attributes.

## Core

### `voxel_editor.py`
- **`create_voxel(voxel_id: Tuple[int, int, int], attributes: Dict[str, float], quantum_state: np.ndarray)`**
  - Creates a voxel with the given attributes and quantum state at the specified position.
- **`edit_voxel(voxel_id: Tuple[int, int, int], new_attributes: Dict[str, float], new_quantum_state: np.ndarray)`**
  - Edits an existing voxel's attributes and quantum state.
- **`get_voxel(voxel_id: Tuple[int, int, int]) -> Voxel`**
  - Retrieves the voxel at the specified position.
- **`delete_voxel(voxel_id: Tuple[int, int, int])`**
  - Deletes the voxel at the specified position.

### `quantum_state_editor.py`
- **`create_quantum_state(voxel_id: Tuple[int, int, int], state_vector: np.ndarray)`**
  - Creates a quantum state for a given voxel.
- **`edit_quantum_state(voxel_id: Tuple[int, int, int], new_state_vector: np.ndarray)`**
  - Edits the quantum state for a voxel.
- **`get_quantum_state(voxel_id: Tuple[int, int, int]) -> np.ndarray`**
  - Retrieves the quantum state for the specified voxel.
- **`delete_quantum_state(voxel_id: Tuple[int, int, int])`**
  - Deletes the quantum state for the specified voxel.

### `attribute_manager.py`
- **`create_attributes(voxel_id: Tuple[int, int, int], attributes: Dict[str, float])`**
  - Creates new attributes for a voxel.
- **`edit_attributes(voxel_id: Tuple[int, int, int], new_attributes: Dict[str, float])`**
  - Edits existing attributes for a voxel.
- **`get_attributes(voxel_id: Tuple[int, int, int]) -> Dict[str, float]`**
  - Retrieves the attributes for the specified voxel.

## Data Management

### `file_loader.py`
- **`load_qvox_file(filename: str) -> Dict[str, np.ndarray]`**
  - Loads a QVOX file and returns the voxel grid and quantum states.
- **`load_voxel_grid(data: Dict[str, np.ndarray]) -> np.ndarray`**
  - Returns the voxel grid from the loaded QVOX data.
- **`load_quantum_states(data: Dict[str, np.ndarray]) -> Dict[Tuple[int, int, int], np.ndarray]`**
  - Returns the quantum states from the loaded QVOX data.

### `file_saver.py`
- **`save_qvox_file(filename: str, voxel_grid: np.ndarray, quantum_states: Dict[Tuple[int, int, int], np.ndarray])`**
  - Saves the voxel grid and quantum states to a QVOX file.

## Rendering

### `voxel_renderer.py`
- **`render_voxel(x: int, y: int, z: int, color_type: str = "default")`**
  - Renders a single voxel at the given coordinates.
- **`render_voxel_grid(voxel_grid: np.ndarray)`**
  - Renders the entire voxel grid in 3D.

### `quantum_renderer.py`
- **`render_quantum_state(x: int, y: int, z: int, quantum_state: np.ndarray)`**
  - Renders a voxel's quantum state in 3D.

## Utilities

### `backup_manager.py`
- **`create_backup(filenames: List[str])`**
  - Creates a backup of the specified QVOX files.
- **`restore_backup(backup_filename: str, restore_dir: str = ".")`**
  - Restores a backup file to the specified directory.

### `logger.py`
- **`log_action(action: str, details: str)`**
  - Logs a specific action with details.

