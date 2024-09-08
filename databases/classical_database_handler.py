# classical_database_handler.py content placeholder
import sqlite3
from typing import List, Tuple, Any

class ClassicalDatabaseHandler:
    def __init__(self, db_name: str):
        """
        Initialize the database handler with the name of the database.
        Connects to the SQLite database (or creates it if it doesn't exist).
        """
        self.db_name = db_name
        self.connection = sqlite3.connect(self.db_name)
        self.cursor = self.connection.cursor()

    def create_table(self, table_name: str, schema: str):
        """
        Create a new table in the database if it doesn't exist.
        schema example: 'id INTEGER PRIMARY KEY, name TEXT, age INTEGER'
        """
        query = f"CREATE TABLE IF NOT EXISTS {table_name} ({schema})"
        self.cursor.execute(query)
        self.connection.commit()

    def insert_record(self, table_name: str, data: Tuple[Any]):
        """
        Insert a new record into the specified table.
        data example: (1, 'Alis', 25) or (NULL, 'Jacob', 30) if id is autoincremented
        """
        placeholders = ', '.join(['?' for _ in data])
        query = f"INSERT INTO {table_name} VALUES ({placeholders})"
        self.cursor.execute(query, data)
        self.connection.commit()

    def insert_multiple_records(self, table_name: str, records: List[Tuple[Any]]):
        """
        Insert multiple records into the specified table in a batch.
        Example: [(NULL, 'Alis', 25), (NULL, 'Jacob', 30)]
        """
        placeholders = ', '.join(['?' for _ in records[0]])
        query = f"INSERT INTO {table_name} VALUES ({placeholders})"
        self.cursor.executemany(query, records)
        self.connection.commit()

    def fetch_all(self, table_name: str) -> List[Tuple]:
        """
        Fetch all records from the specified table.
        """
        query = f"SELECT * FROM {table_name}"
        self.cursor.execute(query)
        return self.cursor.fetchall()

    def fetch_by_id(self, table_name: str, record_id: int) -> Tuple:
        """
        Fetch a specific record by ID from the table.
        """
        query = f"SELECT * FROM {table_name} WHERE id = ?"
        self.cursor.execute(query, (record_id,))
        return self.cursor.fetchone()

    def update_record(self, table_name: str, record_id: int, updated_data: Tuple):
        """
        Update a specific record in the table.
        updated_data should contain values in the same order as they appear in the schema, except for ID.
        """
        columns = self._get_columns(table_name)
        set_clause = ', '.join([f"{col} = ?" for col in columns[1:]])  # Skip 'id'
        query = f"UPDATE {table_name} SET {set_clause} WHERE id = ?"
        self.cursor.execute(query, (*updated_data, record_id))
        self.connection.commit()

    def delete_record(self, table_name: str, record_id: int):
        """
        Delete a specific record by ID from the table.
        """
        query = f"DELETE FROM {table_name} WHERE id = ?"
        self.cursor.execute(query, (record_id,))
        self.connection.commit()

    def _get_columns(self, table_name: str) -> List[str]:
        """
        Fetch the column names of a table.
        """
        self.cursor.execute(f"PRAGMA table_info({table_name})")
        return [col[1] for col in self.cursor.fetchall()]

if __name__ == '__main__':
    # Example usage of the ClassicalDatabaseHandler with Alis and Jacob
    db_handler = ClassicalDatabaseHandler('example.db')

    # Create a table for users
    db_handler.create_table('users', 'id INTEGER PRIMARY KEY, name TEXT, age INTEGER')

    # Insert individual records
    db_handler.insert_record('users', (None, 'Alis', 25))
    db_handler.insert_record('users', (None, 'Jacob', 30))

    # Insert multiple records
    db_handler.insert_multiple_records('users', [(None, 'Alis', 25), (None, 'Jacob', 30)])

    # Fetch all records
    all_records = db_handler.fetch_all('users')
    print("All Records:", all_records)

    # Fetch record by ID
    record = db_handler.fetch_by_id('users', 1)
    print("Record by ID:", record)

    # Update a record (set name to 'Alis Updated' and age to 26)
    db_handler.update_record('users', 1, ('Alis Updated', 26))

    # Delete a record
    db_handler.delete_record('users', 2)

    # Fetch all records after deletion
    updated_records = db_handler.fetch_all('users')
    print("Updated Records:", updated_records)
