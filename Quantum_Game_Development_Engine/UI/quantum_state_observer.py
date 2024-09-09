# quantum_state_observer.py
# Quantum_Game_Development_Engine/UI/quantum_state_observer.py

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

class QuantumStateObserver:
    def __init__(self, master):
        """
        Initializes the QuantumStateObserver with the provided master widget.
        :param master: The parent widget for this observer (usually a root window or frame)
        """
        self.master = master
        self.master.title("Quantum State Observer")

        # Create and configure the main frame
        self.main_frame = tk.Frame(self.master)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create a label for the title
        self.title_label = tk.Label(self.main_frame, text="Quantum State Observer", font=('Arial', 18))
        self.title_label.pack(pady=10)

        # Create a treeview to display quantum states
        self.state_tree = ttk.Treeview(self.main_frame, columns=("State", "Probability"), show='headings')
        self.state_tree.heading("State", text="Quantum State")
        self.state_tree.heading("Probability", text="Probability")
        self.state_tree.pack(fill=tk.BOTH, expand=True, pady=10)

        # Add a scrollbar to the treeview
        self.scrollbar = tk.Scrollbar(self.main_frame, orient=tk.VERTICAL, command=self.state_tree.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.state_tree.configure(yscrollcommand=self.scrollbar.set)

        # Button to refresh quantum state data
        self.refresh_button = tk.Button(self.main_frame, text="Refresh State", command=self.refresh_state, font=('Arial', 14), bg='lightgreen')
        self.refresh_button.pack(pady=10)

        # Initial state data
        self.states = []
        self.load_states()

    def load_states(self):
        """
        Loads initial quantum state data into the observer.
        """
        # For demonstration, pre-populate with some quantum states
        self.states = [
            ("State |0>", "50%"),
            ("State |1>", "50%")
        ]
        self.update_state_display()

    def refresh_state(self):
        """
        Simulates refreshing the quantum state data.
        """
        # Simulate state data retrieval
        self.states = [
            ("State |0>", "45%"),
            ("State |1>", "55%")
        ]
        self.update_state_display()

        # Show a message box
        messagebox.showinfo("Refresh Quantum State", "Quantum state data refreshed!")

    def update_state_display(self):
        """
        Updates the state treeview with the latest quantum state data.
        """
        # Clear the current items
        for item in self.state_tree.get_children():
            self.state_tree.delete(item)

        # Insert new items
        for state, probability in self.states:
            self.state_tree.insert('', tk.END, values=(state, probability))

# Example usage
if __name__ == "__main__":
    root = tk.Tk()
    observer = QuantumStateObserver(root)
    root.mainloop()
