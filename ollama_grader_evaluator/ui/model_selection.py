"""
Model selection component for the Ollama Grader Evaluator.
"""

import tkinter as tk
from tkinter import ttk
from typing import Dict, List


class ModelSelectionFrame(ttk.LabelFrame):
    """
    Frame for selecting models to evaluate.
    """
    
    def __init__(self, parent, models: List[str]):
        """
        Initialize the model selection frame.
        
        Args:
            parent: Parent widget
            models: List of model names to display
        """
        super().__init__(parent, text="Select Models to Evaluate", padding=10)
        
        self.models = models
        self.selected_models = {model: tk.BooleanVar(value=True) for model in models}
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Create the model selection checkboxes."""
        for i, model in enumerate(self.models):
            row, col = divmod(i, 3)
            cb = ttk.Checkbutton(
                self, 
                text=model, 
                variable=self.selected_models[model],
                onvalue=True,
                offvalue=False
            )
            cb.grid(row=row, column=col, sticky=tk.W, padx=10, pady=5)
            
        # Configure grid columns to have equal width
        for i in range(3):  # Assuming 3 columns
            self.columnconfigure(i, weight=1)
    
    def get_selected_models(self) -> List[str]:
        """
        Get the list of selected models.
        
        Returns:
            List of selected model names
        """
        return [model for model, var in self.selected_models.items() if var.get()]
    
    def select_all(self, select=True):
        """
        Select or deselect all models.
        
        Args:
            select: Whether to select (True) or deselect (False) all models
        """
        for var in self.selected_models.values():
            var.set(select)
    
    def add_model(self, model_name: str):
        """
        Add a new model to the selection.
        
        Args:
            model_name: Name of the model to add
        """
        if model_name not in self.selected_models:
            self.models.append(model_name)
            self.selected_models[model_name] = tk.BooleanVar(value=True)
            
            # Recreate the widget to reflect the change
            for widget in self.winfo_children():
                widget.destroy()
            self._create_widgets()
