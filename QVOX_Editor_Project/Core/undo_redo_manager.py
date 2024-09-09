
# undo_redo_manager.py
# Manages undo and redo functionality for voxel editing, quantum state modifications, and attribute changes.

from typing import List, Dict, Tuple

class UndoRedoManager:
    """Class responsible for managing undo and redo stacks for voxel, quantum state, and attribute changes."""

    def __init__(self):
        """Initializes the undo/redo manager with empty stacks."""
        self.undo_stack = []
        self.redo_stack = []

    def add_to_undo(self, action_type: str, voxel_id: Tuple[int, int, int], data_before: Dict, data_after: Dict):
        """Adds an action to the undo stack."""
        self.undo_stack.append({
            "action_type": action_type,
            "voxel_id": voxel_id,
            "data_before": data_before,
            "data_after": data_after
        })
        self.redo_stack.clear()  # Clear redo stack on new action

    def undo(self):
        """Performs the undo operation by reverting the last action."""
        if not self.undo_stack:
            raise IndexError("No actions to undo.")
        last_action = self.undo_stack.pop()
        self.redo_stack.append(last_action)
        return last_action["voxel_id"], last_action["data_before"]

    def redo(self):
        """Performs the redo operation by reapplying the last undone action."""
        if not self.redo_stack:
            raise IndexError("No actions to redo.")
        last_redo = self.redo_stack.pop()
        self.undo_stack.append(last_redo)
        return last_redo["voxel_id"], last_redo["data_after"]

    def clear(self):
        """Clears both undo and redo stacks."""
        self.undo_stack.clear()
        self.redo_stack.clear()

    def get_undo_count(self) -> int:
        """Returns the number of available undo actions."""
        return len(self.undo_stack)

    def get_redo_count(self) -> int:
        """Returns the number of available redo actions."""
        return len(self.redo_stack)

