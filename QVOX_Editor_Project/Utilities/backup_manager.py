
# backup_manager.py
# Manages periodic backups of QVOX files and quantum states, ensuring no data is lost during editing.

import os
import shutil
import time
from typing import List

class BackupManager:
    """Class responsible for managing backups of QVOX files."""

    def __init__(self, backup_dir: str = "backups", backup_interval: int = 300):
        """Initializes the backup manager with a directory and backup interval (in seconds)."""
        self.backup_dir = backup_dir
        self.backup_interval = backup_interval
        os.makedirs(self.backup_dir, exist_ok=True)

    def create_backup(self, filenames: List[str]):
        """Creates a backup of the specified QVOX files."""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        for filename in filenames:
            if os.path.exists(filename):
                backup_filename = os.path.join(self.backup_dir, f"{timestamp}-{os.path.basename(filename)}")
                shutil.copy2(filename, backup_filename)
                print(f"Backup created: {backup_filename}")

    def start_periodic_backup(self, filenames: List[str]):
        """Starts a periodic backup process that runs every backup_interval seconds."""
        while True:
            self.create_backup(filenames)
            time.sleep(self.backup_interval)

    def restore_backup(self, backup_filename: str, restore_dir: str = "."):
        """Restores a backup file to the specified directory."""
        shutil.copy2(backup_filename, restore_dir)
        print(f"Backup restored: {backup_filename}")
