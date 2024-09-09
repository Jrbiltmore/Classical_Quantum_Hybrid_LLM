
# Hexal Format Specification

The Hexal format is designed for representing multidimensional data and quantum states in a structured, grid-based format. This document provides the specifications for the Hexal file format used in Hexal Illustrator.

## Overview

A Hexal file consists of three main sections:

1. **Metadata**: Provides general information about the structure.
2. **Geometry**: Defines the hexagonal grid and its properties.
3. **Attributes**: Specifies attributes like color, opacity, and quantum state information.

## File Structure

A typical Hexal file is represented as a JSON-like structure with the following components:

```json
{
    "name": "HexalStructureName",
    "dimensions": {
        "x": 10,
        "y": 15,
        "z": 5
    },
    "geometry": {
        "type": "hexagonal",
        "grid_size": 5,
        "orientation": "flat-topped",
        "grid_spacing": 2.5
    },
    "attributes": {
        "color": "green",
        "opacity": 0.85,
        "quantum_state": "entangled",
        "temperature": "295K",
        "spin": "1/2"
    },
    "quantum_data": {
        "probability_distribution": [0.25, 0.50, 0.25],
        "entanglement_factor": 0.75
    }
}
```

### Metadata

- **name**: The name of the hexal structure.
- **dimensions**: Specifies the x, y, and z dimensions of the grid. These dimensions define the spatial extent of the structure.

### Geometry

- **type**: Defines the type of grid. Current options include:
  - `hexagonal`: A standard hexagonal grid structure.
- **grid_size**: Specifies the size of each hexagonal tile.
- **orientation**: Determines the layout of the hexagonal grid. Options include:
  - `flat-topped`
  - `pointy-topped`
- **grid_spacing**: Defines the spacing between hexagonal tiles.

### Attributes

- **color**: The color of the hexagonal structure.
- **opacity**: The transparency level, where 1.0 is fully opaque and 0.0 is fully transparent.
- **quantum_state**: Describes the quantum state associated with the structure. Examples include:
  - `superposition`
  - `entangled`
  - `collapsed`
- **temperature**: Specifies the temperature of the structure in Kelvin (K).
- **spin**: Describes the quantum spin of the system (e.g., `1/2`, `1`).

### Quantum Data

- **probability_distribution**: Represents the probability distribution over quantum states.
- **entanglement_factor**: Specifies the degree of entanglement in the quantum system.

## Extensions

The Hexal format is flexible and can be extended with additional attributes or metadata as required by specific applications. For example, you could add a **magnetic_field** attribute to capture the magnetic properties of the structure.

```json
{
    "magnetic_field": "0.01T"
}
```

## Compatibility

Hexal files are compatible with other formats in the Hexal Illustrator (e.g., JSON and CSV) and can be seamlessly converted using the FileConverter module.

For detailed usage, refer to the **Hexal Geometry Tutorial** and **API Reference**.
