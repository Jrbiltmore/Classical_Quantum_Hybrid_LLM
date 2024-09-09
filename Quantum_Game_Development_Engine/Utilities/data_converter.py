# data_converter.py
# Quantum_Game_Development_Engine/Utilities/data_converter.py

import json
import xml.etree.ElementTree as ET
import csv

class DataConverter:
    def __init__(self, input_file, output_file):
        """
        Initializes the DataConverter with input and output file paths.
        :param input_file: Path to the input data file
        :param output_file: Path to the output data file
        """
        self.input_file = input_file
        self.output_file = output_file

    def convert_json_to_csv(self):
        """
        Converts a JSON file to a CSV file.
        """
        with open(self.input_file, 'r') as json_file:
            data = json.load(json_file)
        
        if not isinstance(data, list):
            raise ValueError("JSON data should be a list of dictionaries.")

        with open(self.output_file, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        print(f"Converted JSON to CSV: {self.output_file}")

    def convert_csv_to_json(self):
        """
        Converts a CSV file to a JSON file.
        """
        with open(self.input_file, 'r') as csv_file:
            reader = csv.DictReader(csv_file)
            data = list(reader)
        
        with open(self.output_file, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        print(f"Converted CSV to JSON: {self.output_file}")

    def convert_xml_to_json(self):
        """
        Converts an XML file to a JSON file.
        """
        tree = ET.parse(self.input_file)
        root = tree.getroot()
        data = self._xml_to_dict(root)
        
        with open(self.output_file, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        print(f"Converted XML to JSON: {self.output_file}")

    def convert_json_to_xml(self):
        """
        Converts a JSON file to an XML file.
        """
        with open(self.input_file, 'r') as json_file:
            data = json.load(json_file)
        
        root = ET.Element("root")
        self._dict_to_xml(root, data)
        
        tree = ET.ElementTree(root)
        tree.write(self.output_file)
        print(f"Converted JSON to XML: {self.output_file}")

    def _xml_to_dict(self, element):
        """
        Recursively converts XML elements to a dictionary.
        :param element: XML element to convert
        :return: Dictionary representation of the XML element
        """
        def parse_element(el):
            parsed_data = {}
            for child in el:
                parsed_data[child.tag] = parse_element(child)
            if not parsed_data:
                return el.text
            return parsed_data
        
        return parse_element(element)

    def _dict_to_xml(self, parent, data):
        """
        Recursively converts a dictionary to XML elements.
        :param parent: Parent XML element to append children to
        :param data: Dictionary to convert to XML
        """
        for key, value in data.items():
            if isinstance(value, dict):
                child = ET.SubElement(parent, key)
                self._dict_to_xml(child, value)
            else:
                child = ET.SubElement(parent, key)
                child.text = str(value)

# Example usage
if __name__ == "__main__":
    input_file = "path/to/input_file"
    output_file = "path/to/output_file"
    converter = DataConverter(input_file, output_file)

    # Example conversion operations
    converter.convert_json_to_csv()
    converter.convert_csv_to_json()
    converter.convert_xml_to_json()
    converter.convert_json_to_xml()
