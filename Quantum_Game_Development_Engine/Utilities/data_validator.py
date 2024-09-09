# data_validator.py
# Quantum_Game_Development_Engine/Utilities/data_validator.py

import json
import xml.etree.ElementTree as ET
import csv

class DataValidator:
    def __init__(self, file_path, file_type):
        """
        Initializes the DataValidator with the file path and file type.
        :param file_path: Path to the data file
        :param file_type: Type of the file ('json', 'csv', 'xml')
        """
        self.file_path = file_path
        self.file_type = file_type.lower()

    def validate(self):
        """
        Validates the data file based on the file type.
        :return: True if valid, False otherwise
        """
        if self.file_type == 'json':
            return self._validate_json()
        elif self.file_type == 'csv':
            return self._validate_csv()
        elif self.file_type == 'xml':
            return self._validate_xml()
        else:
            raise ValueError("Unsupported file type. Supported types are: 'json', 'csv', 'xml'.")

    def _validate_json(self):
        """
        Validates a JSON file.
        :return: True if JSON is valid, False otherwise
        """
        try:
            with open(self.file_path, 'r') as file:
                data = json.load(file)
                # Add specific validation checks if needed
                if isinstance(data, dict) or isinstance(data, list):
                    return True
                else:
                    return False
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"JSON validation error: {e}")
            return False

    def _validate_csv(self):
        """
        Validates a CSV file.
        :return: True if CSV is valid, False otherwise
        """
        try:
            with open(self.file_path, 'r') as file:
                reader = csv.reader(file)
                # Check if file has at least one row
                rows = list(reader)
                if rows and all(len(row) > 0 for row in rows):
                    return True
                else:
                    return False
        except FileNotFoundError as e:
            print(f"CSV validation error: {e}")
            return False

    def _validate_xml(self):
        """
        Validates an XML file.
        :return: True if XML is valid, False otherwise
        """
        try:
            tree = ET.parse(self.file_path)
            root = tree.getroot()
            # Add specific validation checks if needed
            if root is not None:
                return True
            else:
                return False
        except ET.ParseError as e:
            print(f"XML validation error: {e}")
            return False

# Example usage
if __name__ == "__main__":
    file_path = "path/to/data_file"
    file_type = "json"  # Could be 'json', 'csv', or 'xml'
    
    validator = DataValidator(file_path, file_type)
    if validator.validate():
        print(f"{file_type.upper()} file is valid.")
    else:
        print(f"{file_type.upper()} file is invalid.")
