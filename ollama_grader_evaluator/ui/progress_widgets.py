"""
Progress display components for the Ollama Grader Evaluator.
"""

import time
import tkinter as tk
from tkinter import ttk, scrolledtext
from typing import Optional


class ProgressFrame(ttk.LabelFrame):
    """
    Frame for displaying evaluation progress.
    """
    
    def __init__(self, parent):
        """
        Initialize the progress display frame.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent, text="Evaluation Progress", padding=10)
        
        # Status variables
        self.progress_var = tk.DoubleVar()
        self.status_var = tk.StringVar(value="Ready")
        self.time_left_var = tk.StringVar(value="")
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Create the progress display widgets."""
        # Progress bar
        self.progress_bar = ttk.Progressbar(
            self, 
            orient=tk.HORIZONTAL, 
            length=500, 
            mode='determinate',
            variable=self.progress_var
        )
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        # Create a frame for status and time estimate
        status_estimate_frame = ttk.Frame(self)
        status_estimate_frame.pack(fill=tk.X, pady=5)
        
        # Status label (left side)
        status_label = ttk.Label(status_estimate_frame, textvariable=self.status_var)
        status_label.pack(side=tk.LEFT, pady=5)
        
        # Time left estimate (right side)
        time_left_label = ttk.Label(status_estimate_frame, textvariable=self.time_left_var)
        time_left_label.pack(side=tk.RIGHT, pady=5)
        
        # Detailed status section
        status_frame = ttk.Frame(self)
        status_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(status_frame, text="Detailed Status:").pack(anchor=tk.W)
        
        # Scrolled text widget for detailed status
        self.detailed_status = scrolledtext.ScrolledText(
            status_frame, 
            height=7, 
            width=80, 
            wrap=tk.WORD, 
            font=("Courier New", 9)
        )
        self.detailed_status.pack(fill=tk.X, pady=5)
        
        # Configure tags for different types of messages
        self.detailed_status.tag_configure("error", foreground="red")
        self.detailed_status.tag_configure("success", foreground="green")
        self.detailed_status.tag_configure("important", foreground="blue", font=("Courier New", 9, "bold"))
    
    def update_progress(self, value: float):
        """
        Update the progress bar.
        
        Args:
            value: Progress value between 0.0 and 1.0
        """
        self.progress_var.set(value * 100)
    
    def update_status(self, status: str):
        """
        Update the status text.
        
        Args:
            status: Status message
        """
        self.status_var.set(status)
    
    def update_time_left(self, time_left: Optional[str]):
        """
        Update the time left estimate.
        
        Args:
            time_left: Time left message or None to clear
        """
        self.time_left_var.set(time_left if time_left else "")
    
    def add_detailed_status(self, message: str, tag: Optional[str] = None):
        """
        Add a message to the detailed status text.
        
        Args:
            message: Status message
            tag: Optional tag for styling ("error", "success", "important")
        """
        timestamp = time.strftime("%H:%M:%S")
        self.detailed_status.insert(tk.END, f"[{timestamp}] {message}\n", tag if tag else "")
        self.detailed_status.see(tk.END)
        self.detailed_status.update_idletasks()  # Force update
        
        # Also print to console for logging
        print(f"[{timestamp}] {message}")
    
    def clear_detailed_status(self):
        """Clear the detailed status text."""
        self.detailed_status.delete(1.0, tk.END)
