
# Getting Started with Hexal Illustrator

Welcome to the Hexal Illustrator! This guide will help you get up and running with the system, providing essential instructions for installation, configuration, and basic usage.

## Prerequisites

Before you start, ensure you have the following installed:

- Python 3.8 or higher
- Pandas library for data manipulation (`pip install pandas`)
- Unity 2021.1 or higher (for Unity integration)
- Git (for version control)
- JSON and CSV file handling libraries

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your_repo/hexal_illustrator.git
   cd hexal_illustrator
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Set up Unity environment for the Unity integration:
   - Open Unity Hub and create a new 3D project.
   - Import the Hexal Illustrator Unity plugin by dragging the `Unity_Assets` folder into the Unity project.
   - Follow the steps in the Unity guide (`Unity_Docs`) to integrate.

## Directory Structure

Here's an overview of the important directories within Hexal Illustrator:

- **Core**: The essential logic of the Hexal Illustrator.
- **Data**: Handles data input/output and storage in various formats (JSON, CSV, Hexal).
- **Docs**: Contains documentation and guides.
- **Hexal_Geometry**: Handles geometric calculations and transformations for hexal structures.
- **Rendering**: Deals with rendering logic for hexal grids and quantum data.
- **Unity_Integration**: Files for integrating the project with Unity for 3D and VR rendering.

## Running the Project

To start using Hexal Illustrator:

1. Run the main Python application:
   ```bash
   python main.py
   ```

2. You can load, edit, and save hexal structures using the data loading and saving modules.

3. For Unity users, launch the Unity project and follow the instructions for rendering hexal grids in real-time.

## Next Steps

For more advanced topics, refer to:

- **API Reference**: For detailed usage of each module.
- **User Guide**: For instructions on working with the Hexal Illustrator features.
- **Hexal Geometry Tutorial**: For in-depth understanding of hexal grids and transformations.
