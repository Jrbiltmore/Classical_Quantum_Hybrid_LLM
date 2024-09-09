
import os
import json
import logging
import pandas as pd
from typing import Any, Dict, Union, Optional

class DataLoader:
    def __init__(self, data_directory: str):
        self.data_directory = data_directory
        self.supported_formats = ['json', 'csv', 'hexal']
        self.logger = logging.getLogger(__name__)
        os.makedirs(self.data_directory, exist_ok=True)

    def load_data(self, file_name: str) -> Union[Dict[str, Any], pd.DataFrame, None]:
        file_path = os.path.join(self.data_directory, file_name)
        if not os.path.exists(file_path):
            self.logger.error(f"File {file_name} does not exist in directory {self.data_directory}")
            return None

        file_extension = file_name.split('.')[-1]
        if file_extension == 'json':
            return self._load_json(file_path)
        elif file_extension == 'csv':
            return self._load_csv(file_path)
        elif file_extension == 'hexal':
            return self._load_hexal(file_path)
        else:
            self.logger.error(f"Unsupported file format: {file_extension}")
            return None

    def _load_json(self, file_path: str) -> Dict[str, Any]:
        try:
            with open(file_path, 'r') as json_file:
                return json.load(json_file)
        except Exception as e:
            self.logger.error(f"Error loading JSON file: {e}")
            return {}

    def _load_csv(self, file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            self.logger.error(f"Error loading CSV file: {e}")
            return pd.DataFrame()

    def _load_hexal(self, file_path: str) -> Optional[Dict[str, Any]]:
        # Placeholder function for loading hexal format
        try:
            # Add the actual logic for reading hexal file format
            self.logger.info(f"Hexal file loading logic needed for {file_path}")
            return {}
        except Exception as e:
            self.logger.error(f"Error loading hexal file: {e}")
            return None

    def save_data(self, file_name: str, data: Union[Dict[str, Any], pd.DataFrame]) -> None:
        file_path = os.path.join(self.data_directory, file_name)
        file_extension = file_name.split('.')[-1]
        
        if file_extension == 'json':
            self._save_json(file_path, data)
        elif file_extension == 'csv':
            self._save_csv(file_path, data)
        elif file_extension == 'hexal':
            self._save_hexal(file_path, data)
        else:
            self.logger.error(f"Unsupported file format for saving: {file_extension}")

    def _save_json(self, file_path: str, data: Dict[str, Any]) -> None:
        try:
            with open(file_path, 'w') as json_file:
                json.dump(data, json_file)
        except Exception as e:
            self.logger.error(f"Error saving JSON file: {e}")

    def _save_csv(self, file_path: str, data: pd.DataFrame) -> None:
        try:
            data.to_csv(file_path, index=False)
        except Exception as e:
            self.logger.error(f"Error saving CSV file: {e}")

    def _save_hexal(self, file_path: str, data: Dict[str, Any]) -> None:
        # Placeholder function for saving hexal format
        try:
            # Add the actual logic for saving hexal file format
            self.logger.info(f"Hexal file saving logic needed for {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving hexal file: {e}")

