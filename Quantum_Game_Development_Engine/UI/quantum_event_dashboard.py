# quantum_event_dashboard.py
# Quantum_Game_Development_Engine/UI/quantum_event_dashboard.py

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

class QuantumEventDashboard:
    def __init__(self, master):
        """
        Initializes the QuantumEventDashboard with the provided master widget.
        :param master: The parent widget for this dashboard (usually a root window or frame)
        """
        self.master = master
        self.master.title("Quantum Event Dashboard")

        # Create and configure the main frame
        self.main_frame = tk.Frame(self.master)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create a label for the title
        self.title_label = tk.Label(self.main_frame, text="Quantum Event Dashboard", font=('Arial', 18))
        self.title_label.pack(pady=10)

        # Create a button to trigger quantum collapse
        self.collapse_button = tk.Button(self.main_frame, text="Trigger Quantum Collapse", command=self.trigger_collapse, font=('Arial', 14), bg='lightblue')
        self.collapse_button.pack(pady=10)

        # Create a treeview to display events
        self.event_tree = ttk.Treeview(self.main_frame, columns=("Event", "Status"), show='headings')
        self.event_tree.heading("Event", text="Event")
        self.event_tree.heading("Status", text="Status")
        self.event_tree.pack(fill=tk.BOTH, expand=True, pady=10)

        # Add a scrollbar to the treeview
        self.scrollbar = tk.Scrollbar(self.main_frame, orient=tk.VERTICAL, command=self.event_tree.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.event_tree.configure(yscrollcommand=self.scrollbar.set)

        # Populate with initial data
        self.events = []
        self.load_events()

    def trigger_collapse(self):
        """
        Simulates a quantum collapse event and updates the event dashboard.
        """
        # Simulate event generation
        event_description = "Quantum Collapse Event"
        event_status = "Triggered"

        # Add event to the dashboard
        self.events.append((event_description, event_status))
        self.update_event_display()

        # Show a message box
        messagebox.showinfo("Quantum Collapse", "Quantum collapse event triggered and logged!")

    def load_events(self):
        """
        Loads initial events into the dashboard.
        """
        # For demonstration, pre-populate with some events
        self.events = [
            ("Quantum Initialization", "Completed"),
            ("Quantum Entanglement", "In Progress")
        ]
        self.update_event_display()

    def update_event_display(self):
        """
        Updates the event treeview with the latest events.
        """
        # Clear the current items
        for item in self.event_tree.get_children():
            self.event_tree.delete(item)

        # Insert new items
        for event_description, event_status in self.events:
            self.event_tree.insert('', tk.END, values=(event_description, event_status))

# Example usage
if __name__ == "__main__":
    root = tk.Tk()
    dashboard = QuantumEventDashboard(root)
    root.mainloop()
