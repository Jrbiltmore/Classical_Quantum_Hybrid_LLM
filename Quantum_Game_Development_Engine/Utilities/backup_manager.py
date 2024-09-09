# backup_manager.py
# Quantum_Game_Development_Engine/Utilities/backup_manager.py

import os
import shutil
import datetime

class BackupManager:
    def __init__(self, data_directory, backup_directory):
        """
        Initializes the BackupManager with data and backup directories.
        :param data_directory: Directory containing game data to be backed up
        :param backup_directory: Directory where backups will be stored
        """
        self.data_directory = data_directory
        self.backup_directory = backup_directory

        # Ensure backup directory exists
        if not os.path.exists(self.backup_directory):
            os.makedirs(self.backup_directory)

    def create_backup(self):
        """
        Creates a backup of the game data directory.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"backup_{timestamp}.zip"
        backup_path = os.path.join(self.backup_directory, backup_name)

        # Create a zip file of the data directory
        shutil.make_archive(backup_path.replace('.zip', ''), 'zip', self.data_directory)
        print(f"Backup created: {backup_path}")

    def restore_backup(self, backup_file):
        """
        Restores game data from a specified backup file.
        :param backup_file: Path to the backup file to restore from
        """
        if not os.path.isfile(backup_file):
            print(f"Backup file does not exist: {backup_file}")
            return
        
        # Extract the zip file to the data directory
        with shutil.ZipFile(backup_file, 'r') as zip_ref:
            zip_ref.extractall(self.data_directory)
        print(f"Backup restored from: {backup_file}")

    def list_backups(self):
        """
        Lists all available backup files in the backup directory.
        """
        backups = [f for f in os.listdir(self.backup_directory) if f.endswith('.zip')]
        return backups

# Example usage
if __name__ == "__main__":
    data_dir = "path/to/game_data"
    backup_dir = "path/to/backups"

    manager = BackupManager(data_dir, backup_dir)

    # Create a backup
    manager.create_backup()

    # List available backups
    backups = manager.list_backups()
    print("Available backups:", backups)

    # Restore a specific backup
    if backups:
        latest_backup = os.path.join(backup_dir, backups[-1])
        manager.restore_backup(latest_backup)
