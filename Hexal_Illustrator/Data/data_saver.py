
import os
import json
import pandas as pd
from typing import Union, Dict, Any

class DataSaver:
    def __init__(self, save_directory: str):
        self.save_directory = save_directory
        os.makedirs(self.save_directory, exist_ok=True)

    def save(self, file_name: str, data: Union[pd.DataFrame, Dict[str, Any]]) -> None:
        file_extension = file_name.split('.')[-1]
        file_path = os.path.join(self.save_directory, file_name)

        if file_extension == 'json':
            self._save_json(file_path, data)
        elif file_extension == 'csv':
            self._save_csv(file_path, data)
        elif file_extension == 'hexal':
            self._save_hexal(file_path, data)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

    def _save_json(self, file_path: str, data: Dict[str, Any]) -> None:
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)

    def _save_csv(self, file_path: str, data: pd.DataFrame) -> None:
        data.to_csv(file_path, index=False)

    def _save_hexal(self, file_path: str, data: Dict[str, Any]) -> None:
        # Placeholder for hexal save logic
        pass
