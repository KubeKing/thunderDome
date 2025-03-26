"""
Main application window for the Ollama Grader Evaluator.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Dict, Any, Optional

from ollama_grader_evaluator.core.data_manager import DataManager
from ollama_grader_evaluator.core.evaluator import Evaluator
from ollama_grader_evaluator.core.export import ResultExporter
from ollama_grader_evaluator.ui.file_input import FileInputFrame
from ollama_grader_evaluator.ui.model_selection import ModelSelectionFrame
from ollama_grader_evaluator.ui.progress_widgets import ProgressFrame
from ollama_grader_evaluator.ui.results_display import ResultsFrame


class MainWindow:
    """
    Main application window for the Ollama Grader Evaluator.
    """
    
    def __init__(self, root):
        """
        Initialize the main window.
        
        Args:
            root: Root Tkinter widget
        """
        self.root = root
        self.root.title("Ollama Grader Evaluator")
        self.root.geometry("1000x800")
        self.root.minsize(800, 600)
        
        # Default model list
        self.models = [
            "gemma3:12b",
            "gemma3:4b",
            "gemma3:1b",
            "qwen2.5:14b",
            "qwen2.5:0.5b",
            "exaone-deep:7.8b",
            "deepseek-r1:14b",
            "deepseek-r1:8b",
            "deepseek-r1:1.5b",
            "mistral:latest",
            "mistral-small:24b"
        ]
        
        # Initialize core components
        self.data_manager = DataManager()
        self.evaluator = Evaluator(update_status_callback=self._update_detailed_status)
        
        # Create the main UI
        self._create_widgets()
    
    def _create_widgets(self):
        """Create the main application widgets."""
        # Create a canvas with scrollbar for scrolling the entire window
        self.canvas = tk.Canvas(self.root)
        self.scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Main frame that will contain all widgets
        main_frame = ttk.Frame(self.canvas, padding=10)
        
        # Create a window in the canvas to hold the main frame
        self.canvas_window = self.canvas.create_window((0, 0), window=main_frame, anchor="nw")
        
        # Configure canvas scrolling
        self.canvas.bind('<Configure>', self._configure_canvas)
        main_frame.bind('<Configure>', self._configure_frame)
        
        # Bind mousewheel scrolling
        self._bind_mousewheel(self.canvas)
        
        # File input section
        self.file_frame = FileInputFrame(
            main_frame, 
            self.data_manager,
            on_file_loaded=self._on_file_loaded
        )
        self.file_frame.pack(fill=tk.X, pady=5)
        
        # Model selection section
        self.model_frame = ModelSelectionFrame(main_frame, self.models)
        self.model_frame.pack(fill=tk.X, pady=5)
        
        # Progress section
        self.progress_frame = ProgressFrame(main_frame)
        self.progress_frame.pack(fill=tk.X, pady=5)
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.run_button = ttk.Button(
            button_frame, 
            text="Run Evaluation", 
            command=self._run_evaluation,
            state=tk.DISABLED
        )
        self.run_button.pack(side=tk.RIGHT, padx=5)
        
        # Results frame (initially empty)
        self.results_frame = ResultsFrame(main_frame)
        self.results_frame.pack(fill=tk.BOTH, expand=True, pady=5)
    
    def _on_file_loaded(self, row_count: int):
        """
        Callback when a file is loaded.
        
        Args:
            row_count: Number of rows in the loaded file
        """
        self.progress_frame.update_status(f"Loaded {row_count} questions")
        self.run_button.configure(state=tk.NORMAL)
    
    def _run_evaluation(self):
        """Start the evaluation process."""
        if self.data_manager.get_data() is None:
            messagebox.showerror("Error", "No data loaded.")
            return
        
        selected_models = self.model_frame.get_selected_models()
        
        if not selected_models:
            messagebox.showerror("Error", "Please select at least one model.")
            return
        
        # Disable run button during evaluation
        self.run_button.configure(state=tk.DISABLED)
        
        # Clear progress display
        self.progress_frame.clear_detailed_status()
        self.progress_frame.add_detailed_status("Starting evaluation...", "important")
        
        # Reset progress
        self.progress_frame.update_progress(0)
        
        # Start evaluation
        self.evaluator.evaluate(
            data=self.data_manager.get_data(),
            selected_models=selected_models,
            on_progress=self._on_evaluation_progress,
            on_complete=self._on_evaluation_complete
        )
    
    def _on_evaluation_progress(self, progress: float, status: str, time_left: Optional[str]):
        """
        Callback for evaluation progress updates.
        
        Args:
            progress: Progress value between 0.0 and 1.0
            status: Status message
            time_left: Estimated time remaining
        """
        self.progress_frame.update_progress(progress)
        self.progress_frame.update_status(status)
        self.progress_frame.update_time_left(time_left)
    
    def _update_detailed_status(self, message: str, tag: Optional[str] = None):
        """
        Update the detailed status with a message.
        
        Args:
            message: Status message
            tag: Optional tag for styling
        """
        self.progress_frame.add_detailed_status(message, tag)
    
    def _on_evaluation_complete(self):
        """Callback when evaluation is complete."""
        # This callback is invoked from a non-main thread
        # Use after() to schedule UI updates in the main thread
        self.root.after(0, self._update_ui_after_evaluation_complete)
    
    def _update_ui_after_evaluation_complete(self):
        """Update UI elements after evaluation completes, called in main thread."""
        self.progress_frame.update_status("Evaluation complete!")
        self.run_button.configure(state=tk.NORMAL)
        
        # Get results and display them
        results = self.evaluator.get_results()
        
        self.results_frame.display_results(
            results=results,
            on_export=lambda: self._export_results(results),
            on_export_csv=lambda: self._export_as_csv(results)
        )
    
    def _export_results(self, results: Dict[str, List[Dict[str, Any]]]):
        """
        Export all results in various formats.
        
        Args:
            results: Dictionary of results by model
        """
        exporter = ResultExporter(results)
        export_paths = exporter.export_all()
        
        if export_paths:
            paths_text = "\n- ".join(export_paths.values())
            export_msg = f"Results exported to:\n- {paths_text}"
            self.progress_frame.add_detailed_status(export_msg, "success")
            messagebox.showinfo("Export Complete", export_msg)
    
    def _export_as_csv(self, results: Dict[str, List[Dict[str, Any]]]):
        """
        Export detailed results to CSV files.
        
        Args:
            results: Dictionary of results by model
        """
        exporter = ResultExporter(results)
        
        try:
            detailed_csv_path = exporter.export_detailed_csv(
                detailed_path="evaluation_detailed.csv",
                responses_path="evaluation_full_responses.csv"
            )
            
            export_msg = f"Detailed results exported to CSV files"
            self.progress_frame.add_detailed_status(export_msg, "success")
            messagebox.showinfo("Export Complete", export_msg)
                
        except Exception as e:
            error_msg = f"Failed to export CSV: {str(e)}"
            self.progress_frame.add_detailed_status(error_msg, "error")
            messagebox.showerror("Export Error", error_msg)
    
    def _configure_canvas(self, event):
        """
        Configure the canvas scrolling region when the canvas is resized.
        
        Args:
            event: The Configure event
        """
        # Update the scrollable region to encompass the inner frame
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
        # Set the canvas width to match the window width
        self.canvas.itemconfig(self.canvas_window, width=event.width)
    
    def _configure_frame(self, event):
        """
        Adjust the canvas scrolling region when the inner frame size changes.
        
        Args:
            event: The Configure event
        """
        # Update the scrollable region to match the size of the inner frame
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def _bind_mousewheel(self, widget):
        """
        Bind mousewheel events to the widget for scrolling.
        
        Args:
            widget: The widget to bind mousewheel events to
        """
        # Bind for Windows/Linux
        widget.bind("<MouseWheel>", self._on_mousewheel)
        # Bind for Linux
        widget.bind("<Button-4>", self._on_mousewheel)
        widget.bind("<Button-5>", self._on_mousewheel)
    
    def _on_mousewheel(self, event):
        """
        Handle mousewheel events for scrolling.
        
        Args:
            event: The mousewheel event
        """
        # Cross-platform mousewheel scrolling
        if event.num == 4 or event.delta > 0:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5 or event.delta < 0:
            self.canvas.yview_scroll(1, "units")
