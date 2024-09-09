
# observer_data_handler.py
# Stores and manages observer-related data in QVOX files.

import json
from typing import Dict, Tuple

class ObserverDataHandler:
    """Class responsible for managing and saving observer-related data."""

    def save_observer_data(self, filename: str, observer_data: Dict[str, float]):
        """Saves observer-related data (e.g., position, velocity) to a QVOX file."""
        with open(filename, 'w') as file:
            json.dump(observer_data, file, indent=4)

    def load_observer_data(self, filename: str) -> Dict[str, float]:
        """Loads observer-related data from a QVOX file."""
        with open(filename, 'r') as file:
            observer_data = json.load(file)
        return observer_data
