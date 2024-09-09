
# attribute_viewer.py
# UI component to display and edit the multidimensional attributes of voxels.

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QLineEdit, QFormLayout, QPushButton
from typing import Dict, Tuple

class AttributeViewer(QWidget):
    """Widget for displaying and editing the multidimensional attributes of a voxel."""

    def __init__(self, parent=None):
        super(AttributeViewer, self).__init__(parent)
        self.layout = QVBoxLayout(self)
        self.attribute_data = {}  # Stores the current voxel's attribute data
        self.attribute_editors = {}  # Stores the QLineEdit widgets for each attribute

        # Create form layout for displaying attributes
        self.form_layout = QFormLayout()
        self.layout.addLayout(self.form_layout)

        # Label to show which voxel is being edited
        self.voxel_label = QLabel("No Voxel Selected")
        self.layout.addWidget(self.voxel_label)

        # Save button
        self.save_button = QPushButton("Save Attributes")
        self.save_button.clicked.connect(self.save_attributes)
        self.layout.addWidget(self.save_button)

    def load_attributes(self, voxel_id: Tuple[int, int, int], attributes: Dict[str, float]):
        """Loads the attributes of a voxel for editing."""
        self.voxel_label.setText(f"Editing Voxel: {voxel_id}")
        self.attribute_data = attributes

        # Clear existing form fields
        for editor in self.attribute_editors.values():
            editor.deleteLater()
        self.attribute_editors.clear()

        # Create new form fields for each attribute
        for attr_name, attr_value in attributes.items():
            editor = QLineEdit(str(attr_value))
            self.attribute_editors[attr_name] = editor
            self.form_layout.addRow(attr_name, editor)

    def save_attributes(self):
        """Saves the edited attributes back to the attribute manager."""
        for attr_name, editor in self.attribute_editors.items():
            try:
                new_value = float(editor.text())
                self.attribute_data[attr_name] = new_value
            except ValueError:
                pass  # Ignore invalid input for now

        # Update the attribute manager (this would call the manager in a real application)
        print(f"Updated attributes: {self.attribute_data}")

