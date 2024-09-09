# Hexal Illustrator

Hexal Illustrator is a sophisticated tool designed for rendering, visualizing, and manipulating hexagonal structures (hexals) in both classical and quantum dimensions. It supports dynamic observer-based rendering, multidimensional attributes, real-time updates, and interaction with quantum state data.

## Features

- **Hexal Rendering**: Visualize and interact with hexagonal grids.
- **Quantum Data Integration**: Apply and visualize quantum states in the hexal grid.
- **Multidimensional Attributes**: Handle quantum entanglement, entropy, spin, and other multidimensional properties.
- **Observer-Based Dynamics**: Adjust rendering based on the observer’s perspective and speed, with support for relativistic effects.
- **Real-Time Updates**: Integrate real-time feedback and update the UI dynamically.
- **Unity Integration**: Support for Unity assets, shaders, and scene integration for enhanced 3D rendering and interactions.

## Directory Structure

The project is organized into several key components:

```
/Hexal_Illustrator
  ├── /Core                 # Core logic and data handling for hexal rendering and quantum integration
  ├── /UI                   # User interface components, including hexal and quantum viewers, attribute panels, and more
  ├── /Data                 # Data management for saving, loading, and exporting hexal grids
  ├── /Rendering            # Advanced rendering logic for visualizing hexal grids, quantum states, and multidimensional attributes
  ├── /Hexal_Geometry       # Geometric manipulation of the hexal grid (e.g., scaling, rotating)
  ├── /Multidimensional_Attributes   # Quantum and multidimensional attribute management (spin, entropy, entanglement)
  ├── /Observer_Dynamics     # Observer-based adjustments to visualization, simulating quantum and relativistic effects
  ├── /Quantum_States        # Handling quantum state management, including superpositions, collapses, and wavefunction evolution
  ├── /Testing              # Unit and integration tests for validation
  ├── /Utilities            # Backup, logging, and data conversion tools
  ├── /Plugins              # Additional functionality and external integration, including Quantum SQL and Unity
  ├── /Unity_Integration    # Unity asset, shader, and scene integration for enhanced 3D and VR rendering
```

## Requirements

- Python 3.x
- PyQt5 (for the UI)
- NumPy (for numerical operations)
- Matplotlib (for rendering)
- Unity (optional, for 3D and VR integration)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/jrbiltmore/hexal-illustrator.git
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Set up Unity integration by downloading and importing the required Unity assets.

## Usage

1. Run the main application:
   ```bash
   python main.py
   ```

2. Use the UI to:
   - Draw, erase, and modify hexal cells.
   - Visualize quantum states and multidimensional attributes.
   - Adjust the observer's perspective and apply real-time updates.

3. (Optional) For Unity integration, follow the instructions in the `/Unity_Docs` folder to set up scenes and assets.

## Key Modules

- **Hexal Engine**: Core logic for initializing, updating, and manipulating the hexagonal grid.
- **Quantum Data Integration**: Manages quantum states and multidimensional attributes, including quantum collapses and entanglement.
- **UI Components**: Provides the graphical interface for user interactions, rendering, and data visualization.
- **Observer Dynamics**: Adjusts the hexal grid visualization based on the observer's movement and view, simulating relativity effects.
- **Unity Integration**: Allows you to export hexal grids to Unity for enhanced 3D and VR experiences.

## License

This project is licensed under the MIT License.
