"""
Data loading and processing for the Ollama Grader Evaluator.
"""

import pandas as pd
from typing import Optional, List, Dict, Any
from tkinter import messagebox


class DataManager:
    """
    Manages loading and validation of evaluation data from files.
    """
    
    def __init__(self):
        """Initialize the DataManager."""
        self.data = None
        self.required_columns = ["Question", "Model Answer", "Student Answer", "Model Grade"]
    
    def load_file(self, file_path: str) -> bool:
        """
        Load data from a CSV or Excel file.
        
        Args:
            file_path: Path to the file to load
            
        Returns:
            True if the file was loaded successfully, False otherwise
        """
        try:
            if file_path.endswith(('.xlsx', '.xls')):
                self.data = pd.read_excel(file_path)
            else:
                self.data = pd.read_csv(file_path)
            
            return self.validate_data()
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")
            self.data = None
            return False
    
    def validate_data(self) -> bool:
        """
        Validate that the loaded data has the required columns.
        
        Returns:
            True if the data is valid, False otherwise
        """
        if self.data is None:
            return False
            
        missing_columns = [col for col in self.required_columns if col not in self.data.columns]
        
        if missing_columns:
            messagebox.showerror(
                "Missing Columns", 
                f"The following required columns are missing: {', '.join(missing_columns)}"
            )
            self.data = None
            return False
            
        return True
    
    def get_data(self) -> Optional[pd.DataFrame]:
        """
        Get the loaded data.
        
        Returns:
            The loaded DataFrame, or None if no data is loaded
        """
        return self.data
    
    def get_row_count(self) -> int:
        """
        Get the number of rows in the loaded data.
        
        Returns:
            Number of rows, or 0 if no data is loaded
        """
        return len(self.data) if self.data is not None else 0
