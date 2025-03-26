"""
Results visualization components for the Ollama Grader Evaluator.
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import Dict, List, Any, Optional, Callable


class ResultsFrame(ttk.LabelFrame):
    """
    Frame for displaying evaluation results.
    """
    
    def __init__(self, parent):
        """
        Initialize the results display frame.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent, text="Results", padding=10)
        
        # Frame to contain the canvas and other widgets
        self.canvas_frame = ttk.Frame(self)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
    
    def display_results(self, results: Dict[str, List[Dict[str, Any]]], 
                       on_export: Optional[Callable] = None,
                       on_export_csv: Optional[Callable] = None):
        """
        Display evaluation results with charts and tables.
        
        Args:
            results: Dictionary of results by model
            on_export: Callback for exporting results
            on_export_csv: Callback for exporting as CSV
        """
        # Clear previous results
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()
        
        # Calculate metrics by model - this is safe to do in any thread
        model_metrics = self._calculate_model_metrics(results)
        
        # Schedule the UI update in the main thread using after()
        self.after(10, lambda: self._render_results_in_main_thread(model_metrics, on_export, on_export_csv))
    
    def _render_results_in_main_thread(self, model_metrics: Dict[str, Dict[str, Any]], 
                                      on_export: Optional[Callable] = None,
                                      on_export_csv: Optional[Callable] = None):
        """
        Render the results in the main thread to avoid threading issues with matplotlib.
        This method should only be called from the main thread.
        
        Args:
            model_metrics: Dictionary of metrics by model
            on_export: Callback for exporting results
            on_export_csv: Callback for exporting as CSV
        """
        # Extract data for plotting
        models = list(model_metrics.keys())
        accuracies = [model_metrics[model]["accuracy"] for model in models]
        consistencies = [model_metrics[model]["consistency"] for model in models]
        times = [model_metrics[model]["response_time"] for model in models]
        
        # Create figure for plotting
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))
        
        # Plot 1: Average accuracy
        bars1 = ax1.bar(models, accuracies, color='skyblue')
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Average Accuracy')
        ax1.set_title('Average Grading Accuracy by Model')
        ax1.set_ylim(0, 1)
        
        # Fix for set_ticklabels warning: set the ticks first
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels(models, rotation=45, ha='right')
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom', rotation=0)
        
        # Plot 2: Consistency
        bars2 = ax2.bar(models, consistencies, color='#FFB6C1')  # Light pink
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Consistency Score')
        ax2.set_title('Grading Consistency by Model')
        ax2.set_ylim(0, 1)
        
        # Fix for set_ticklabels warning: set the ticks first
        ax2.set_xticks(range(len(models)))
        ax2.set_xticklabels(models, rotation=45, ha='right')
        
        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom', rotation=0)
        
        # Plot 3: Average response time
        bars3 = ax3.bar(models, times, color='lightgreen')
        ax3.set_xlabel('Model')
        ax3.set_ylabel('Average Response Time (s)')
        ax3.set_title('Average Response Time by Model')
        
        # Fix for set_ticklabels warning: set the ticks first
        ax3.set_xticks(range(len(models)))
        ax3.set_xticklabels(models, rotation=45, ha='right')
        
        # Add value labels
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}s', ha='center', va='bottom', rotation=0)
        
        fig.tight_layout()
        
        # Add to canvas - this must happen in the main thread
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add results table
        self._add_results_table(model_metrics)
        
        # Add export buttons if callbacks are provided
        if on_export or on_export_csv:
            export_frame = ttk.Frame(self.canvas_frame)
            export_frame.pack(pady=10)
            
            if on_export:
                export_button = ttk.Button(
                    export_frame, 
                    text="Export Results", 
                    command=on_export
                )
                export_button.pack(side=tk.LEFT, padx=5)
            
            if on_export_csv:
                export_csv_button = ttk.Button(
                    export_frame, 
                    text="Export as CSV", 
                    command=on_export_csv
                )
                export_csv_button.pack(side=tk.LEFT, padx=5)
            
    def _add_results_table(self, model_metrics: Dict[str, Dict[str, Any]]):
        """
        Add a table displaying summary results.
        
        Args:
            model_metrics: Dictionary of metrics by model
        """
        # Create frame for table
        table_frame = ttk.Frame(self.canvas_frame)
        table_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create table headers
        columns = ["Model", "Avg Accuracy", "Consistency", "Avg Response Time", "Extraction Confidence"]
        
        for i, col in enumerate(columns):
            ttk.Label(table_frame, text=col, font=("TkDefaultFont", 10, "bold")).grid(
                row=0, column=i, padx=5, pady=5, sticky=tk.W
            )
        
        # Add data rows
        for i, (model, stats) in enumerate(model_metrics.items()):
            ttk.Label(table_frame, text=model).grid(
                row=i+1, column=0, padx=5, pady=2, sticky=tk.W
            )
            
            ttk.Label(table_frame, text=f"{stats['accuracy']:.3f}").grid(
                row=i+1, column=1, padx=5, pady=2
            )
            
            ttk.Label(table_frame, text=f"{stats['consistency']:.3f}").grid(
                row=i+1, column=2, padx=5, pady=2
            )
            
            ttk.Label(table_frame, text=f"{stats['response_time']:.2f}s").grid(
                row=i+1, column=3, padx=5, pady=2
            )
            
            ttk.Label(table_frame, text=f"{stats['confidence']}").grid(
                row=i+1, column=4, padx=5, pady=2
            )
            
        # Configure grid to expand properly
        for i in range(len(columns)):
            table_frame.columnconfigure(i, weight=1)
    
    def _calculate_model_metrics(self, results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
        """
        Calculate summary metrics for each model.
        
        Args:
            results: Dictionary of results by model
            
        Returns:
            Dictionary of metrics by model
        """
        model_metrics = {}
        
        for model, model_results in results.items():
            # Group results by question to calculate per-question metrics
            questions = {}
            for r in model_results:
                q_id = r["question_id"]
                if q_id not in questions:
                    questions[q_id] = []
                questions[q_id].append(r)
            
            # Calculate average accuracy across all attempts
            accuracies = [r.get("accuracy", 0) for r in model_results]
            avg_accuracy = np.mean(accuracies) if accuracies else 0
            
            # Calculate average consistency score
            consistency_scores = [r.get("consistency_score", 0) for r in model_results if "consistency_score" in r]
            avg_consistency = np.mean(consistency_scores) if consistency_scores else 0
            
            # Calculate average response time
            times = [r.get("response_time", 0) for r in model_results]
            avg_time = np.mean(times) if times else 0
            
            # Determine most common confidence level
            confidences = [r.get("confidence", "low") for r in model_results if "confidence" in r]
            confidence_counts = {
                "high": 0,
                "medium": 0,
                "low": 0,
                "very low": 0
            }
            
            for conf in confidences:
                if conf in confidence_counts:
                    confidence_counts[conf] += 1
            
            # Determine most common confidence
            most_common = max(confidence_counts.items(), key=lambda x: x[1]) if confidence_counts else ("unknown", 0)
            confidence_text = most_common[0]
            
            model_metrics[model] = {
                "accuracy": avg_accuracy,
                "consistency": avg_consistency,
                "response_time": avg_time,
                "confidence": confidence_text,
                "questions": len(questions)
            }
        
        return model_metrics
