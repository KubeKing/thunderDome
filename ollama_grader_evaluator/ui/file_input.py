"""
File input components for the Ollama Grader Evaluator.
"""

import tkinter as tk
from tkinter import ttk, filedialog

from ollama_grader_evaluator.core.data_manager import DataManager


class FileInputFrame(ttk.LabelFrame):
    """
    Frame for file selection and loading.
    """
    
    def __init__(self, parent, data_manager: DataManager, on_file_loaded=None):
        """
        Initialize the file input frame.
        
        Args:
            parent: Parent widget
            data_manager: DataManager instance for handling file data
            on_file_loaded: Callback function called when a file is successfully loaded
        """
        super().__init__(parent, text="Input Data", padding=10)
        
        self.data_manager = data_manager
        self.on_file_loaded = on_file_loaded
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Create the file input widgets."""
        ttk.Label(self, text="Select CSV or Excel file:").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        self.file_path_var = tk.StringVar()
        file_path_entry = ttk.Entry(self, textvariable=self.file_path_var, width=50)
        file_path_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        browse_button = ttk.Button(self, text="Browse", command=self._browse_file)
        browse_button.grid(row=0, column=2, padx=5, pady=5)
        
        # Configure grid to expand properly
        self.columnconfigure(1, weight=1)
    
    def _browse_file(self):
        """Open file browser dialog and load the selected file."""
        filetypes = [
            ("CSV files", "*.csv"),
            ("Excel files", "*.xlsx *.xls"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select a file",
            filetypes=filetypes
        )
        
        if filename:
            self.file_path_var.set(filename)
            self._load_file(filename)
    
    def _load_file(self, file_path):
        """
        Load the file and notify the parent if successful.
        
        Args:
            file_path: Path to the file to load
        """
        if self.data_manager.load_file(file_path):
            row_count = self.data_manager.get_row_count()
            if self.on_file_loaded:
                self.on_file_loaded(row_count)
