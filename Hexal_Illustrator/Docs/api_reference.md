
# Hexal Illustrator API Reference

This document provides a comprehensive API reference for the Hexal Illustrator, covering all core classes, functions, and modules.

## Core Modules

### DataLoader

- **`__init__(data_directory: str)`**  
  Initializes the DataLoader with a specified directory for loading data.

- **`load_data(file_name: str) -> Union[Dict[str, Any], pd.DataFrame, None]`**  
  Loads data from a file. Supports JSON, CSV, and Hexal formats.

- **`save_data(file_name: str, data: Union[Dict[str, Any], pd.DataFrame]) -> None`**  
  Saves data to a file. Supports JSON, CSV, and Hexal formats.

---

### DataSaver

- **`__init__(save_directory: str)`**  
  Initializes the DataSaver with a specified directory for saving data.

- **`save(file_name: str, data: Union[pd.DataFrame, Dict[str, Any]]) -> None`**  
  Saves data to a file in the specified format (JSON, CSV, or Hexal).

---

### FileConverter

- **`__init__(source_directory: str, target_directory: str)`**  
  Initializes the FileConverter with directories for source and target files.

- **`convert(source_file: str, target_format: str) -> None`**  
  Converts a file from one format to another (JSON, CSV, or Hexal).

---

### ExportManager

- **`__init__(export_directory: str)`**  
  Initializes the ExportManager with a specified directory for exporting data.

- **`export_data(file_name: str, data: Union[Dict[str, Any], pd.DataFrame], format: str) -> None`**  
  Exports data to a specified format (JSON, CSV, or Hexal).

- **`list_exported_files() -> list`**  
  Lists all exported files in the directory.

---

## Hexal Geometry

### HexalGrid

- **`generate_grid(dimensions: Tuple[int, int], grid_size: int, orientation: str) -> np.ndarray`**  
  Generates a hexagonal grid based on the specified dimensions, grid size, and orientation.

- **`transform_grid(grid: np.ndarray, transformation: str) -> np.ndarray`**  
  Applies a transformation to the hexal grid (e.g., rotation, scaling).

---

### HexalTransformations

- **`rotate(grid: np.ndarray, angle: float) -> np.ndarray`**  
  Rotates the hexal grid by a specified angle.

- **`scale(grid: np.ndarray, factor: float) -> np.ndarray`**  
  Scales the hexal grid by a specified factor.

---

## Rendering

### HexalRenderer

- **`render_grid(grid: np.ndarray) -> None`**  
  Renders the hexagonal grid in 2D or 3D.

- **`render_quantum_data(grid: np.ndarray, quantum_data: dict) -> None`**  
  Visualizes quantum state data on the hexal grid.

---

## Unity Integration

### QuantumIntegrationController (Unity)

- **`initialize_quantum_states()`**  
  Initializes quantum states for real-time rendering in Unity.

- **`update_hexal_grid()`**  
  Updates the hexal grid dynamically within the Unity environment.

---

### ObserverUnityAdapter (Unity)

- **`adapt_observer_view(view_data: dict)`**  
  Adapts the observer's perspective in Unity to account for changes in quantum state visualization.

---

For detailed examples and usage, refer to the **User Guide** and individual module documentation.

