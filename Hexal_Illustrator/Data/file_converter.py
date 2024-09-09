
import os
import pandas as pd
import json
from typing import Union, Dict, Any

class FileConverter:
    def __init__(self, source_directory: str, target_directory: str):
        self.source_directory = source_directory
        self.target_directory = target_directory
        os.makedirs(self.target_directory, exist_ok=True)

    def convert(self, source_file: str, target_format: str) -> None:
        source_path = os.path.join(self.source_directory, source_file)
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source file {source_file} does not exist.")
        
        source_extension = source_file.split('.')[-1]
        data = self._load_file(source_path, source_extension)
        self._save_converted_file(source_file, data, target_format)

    def _load_file(self, file_path: str, file_format: str) -> Union[pd.DataFrame, Dict[str, Any]]:
        if file_format == 'json':
            with open(file_path, 'r') as f:
                return json.load(f)
        elif file_format == 'csv':
            return pd.read_csv(file_path)
        elif file_format == 'hexal':
            return self._load_hexal(file_path)
        else:
            raise ValueError(f"Unsupported source file format: {file_format}")

    def _save_converted_file(self, source_file: str, data: Union[pd.DataFrame, Dict[str, Any]], target_format: str) -> None:
        file_name = source_file.split('.')[0]
        target_file = f"{file_name}.{target_format}"
        target_path = os.path.join(self.target_directory, target_file)

        if target_format == 'json':
            with open(target_path, 'w') as f:
                json.dump(data, f, indent=4)
        elif target_format == 'csv':
            data.to_csv(target_path, index=False)
        elif target_format == 'hexal':
            self._save_hexal(target_path, data)
        else:
            raise ValueError(f"Unsupported target file format: {target_format}")

    def _load_hexal(self, file_path: str) -> Dict[str, Any]:
        # Placeholder for loading hexal format
        pass

    def _save_hexal(self, file_path: str, data: Dict[str, Any]) -> None:
        # Placeholder for saving hexal format
        pass
