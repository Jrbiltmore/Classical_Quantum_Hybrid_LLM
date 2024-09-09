
import os
import shutil
import time

class VersionControl:
    def __init__(self, version_dir="qvox_versions"):
        self.version_dir = version_dir
        os.makedirs(self.version_dir, exist_ok=True)

    def save_version(self, filename):
        """Saves a version of the current QVOX file."""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        version_filename = os.path.join(self.version_dir, f"{os.path.basename(filename)}_{timestamp}.qvox")
        shutil.copy2(filename, version_filename)
        print(f"Version saved: {version_filename}")
        return version_filename

    def list_versions(self, base_filename):
        """Lists all versions of the given QVOX file."""
        versions = [f for f in os.listdir(self.version_dir) if f.startswith(base_filename)]
        return versions

    def restore_version(self, version_filename, restore_to_filename):
        """Restores a previous version of the QVOX file."""
        shutil.copy2(os.path.join(self.version_dir, version_filename), restore_to_filename)
        print(f"Restored {version_filename} to {restore_to_filename}.")
