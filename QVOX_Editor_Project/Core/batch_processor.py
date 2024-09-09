
import os
from file_converter import FileConverter

class BatchProcessor:
    def __init__(self, directory):
        self.directory = directory
        self.converter = FileConverter()

    def process_qvox_files(self, operation):
        """Processes all QVOX files in the specified directory with the given operation."""
        for filename in os.listdir(self.directory):
            if filename.endswith(".qvox"):
                file_path = os.path.join(self.directory, filename)
                print(f"Processing file: {file_path}")
                operation(file_path)

    def convert_all_to_hdf5(self, output_dir):
        """Batch converts all QVOX files to HDF5 format."""
        os.makedirs(output_dir, exist_ok=True)
        for filename in os.listdir(self.directory):
            if filename.endswith(".qvox"):
                qvox_file = os.path.join(self.directory, filename)
                hdf5_file = os.path.join(output_dir, filename.replace(".qvox", ".hdf5"))
                self.converter.qvox_to_hdf5(qvox_file, hdf5_file)
