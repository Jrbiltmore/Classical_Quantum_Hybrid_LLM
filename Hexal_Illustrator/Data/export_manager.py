
import os
import pandas as pd
import json
from typing import Dict, Any, Union

class ExportManager:
    def __init__(self, export_directory: str):
        self.export_directory = export_directory
        os.makedirs(self.export_directory, exist_ok=True)

    def export_data(self, file_name: str, data: Union[Dict[str, Any], pd.DataFrame], format: str) -> None:
        file_extension = format.lower()
        file_path = os.path.join(self.export_directory, f"{file_name}.{file_extension}")
        
        if file_extension == 'json':
            self._export_json(file_path, data)
        elif file_extension == 'csv':
            self._export_csv(file_path, data)
        elif file_extension == 'hexal':
            self._export_hexal(file_path, data)
        else:
            raise ValueError(f"Unsupported export format: {file_extension}")

    def _export_json(self, file_path: str, data: Dict[str, Any]) -> None:
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            raise RuntimeError(f"Failed to export JSON: {str(e)}")

    def _export_csv(self, file_path: str, data: pd.DataFrame) -> None:
        try:
            data.to_csv(file_path, index=False)
        except Exception as e:
            raise RuntimeError(f"Failed to export CSV: {str(e)}")

    def _export_hexal(self, file_path: str, data: Dict[str, Any]) -> None:
        # Placeholder for exporting to hexal format
        pass

    def list_exported_files(self) -> list:
        try:
            return os.listdir(self.export_directory)
        except Exception as e:
            raise RuntimeError(f"Failed to list files: {str(e)}")
