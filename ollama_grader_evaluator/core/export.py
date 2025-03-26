"""
Export functionality for the Ollama Grader Evaluator.
"""

import os
import json
import pandas as pd
from typing import Dict, Any, List, Optional
from tkinter import messagebox
import numpy as np # Added import for numpy

from ollama_grader_evaluator.utils.helpers import generate_timestamp_filename


class ResultExporter:
    """
    Handles exporting evaluation results in various formats.
    """
    
    def __init__(self, results: Dict[str, List[Dict[str, Any]]]):
        """
        Initialize the exporter with results data.
        
        Args:
            results: Dictionary of results, keyed by model name
        """
        self.results = results
    
    def export_all(self, base_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Export results in all supported formats.
        
        Args:
            base_dir: Directory to save files (defaults to script directory)
            
        Returns:
            Dictionary of format to file path
        """
        try:
            # Use script directory if none provided
            if base_dir is None:
                base_dir = os.path.dirname(os.path.abspath(__file__))
                # Move up from core directory to main package directory
                base_dir = os.path.dirname(os.path.dirname(base_dir))

            # Define and create exports directory
            export_dir = os.path.join(base_dir, "exports")
            os.makedirs(export_dir, exist_ok=True)

            # Generate paths for CSV formats within the exports directory
            csv_path = os.path.join(export_dir, generate_timestamp_filename("evaluation_results", "csv"))
            detailed_csv_path = os.path.join(export_dir, generate_timestamp_filename("evaluation_detailed", "csv"))
            responses_path = os.path.join(export_dir, generate_timestamp_filename("evaluation_full_responses", "csv"))

            # Export in CSV formats
            self.export_to_csv(csv_path)
            self.export_detailed_csv(detailed_csv_path, responses_path)

            return {
                "csv": csv_path,
                "detailed_csv": detailed_csv_path,
                "responses_csv": responses_path,
            }
                
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export results: {str(e)}")
            return {}

    def export_to_csv(self, path: str) -> None:
        """
        Export a summary of all results to a CSV file.
        
        Args:
            path: Path to save the CSV file
        """
        # For CSV, we'll just export a summary of all models
        all_results = []
        for model, results in self.results.items():
            for result in results:
                result_copy = result.copy()
                result_copy['model'] = model
                all_results.append(result_copy)
        
        df = pd.DataFrame(all_results)
        # Ensure columns exist before selecting
        cols_to_keep = [col for col in ['model', 'question_id', 'question', 'attempt', 'model_grade', 'extracted_grade', 'accuracy', 'consistency_score', 'grade_stability', 'response_time', 'confidence'] if col in df.columns]
        other_cols = [col for col in df.columns if col not in cols_to_keep]
        df = df[cols_to_keep + other_cols] # Reorder if needed, keep all columns for summary CSV
        df.to_csv(path, index=False)

    def export_detailed_csv(self, detailed_path: str, responses_path: str) -> None:
        """
        Export detailed results to CSV files, including one with full responses.
        
        Args:
            detailed_path: Path to save the detailed CSV file
            responses_path: Path to save the full responses CSV file
        """
        # Prepare data with all details
        all_results = []
        for model, results in self.results.items():
            for result in results:
                # Create a copy to avoid modifying the original
                result_copy = result.copy()
                result_copy['model'] = model
                
                # Truncate long texts for CSV readability
                if 'full_response' in result_copy:
                    # Keep full response but prepare a truncated version for summary
                    result_copy['response_snippet'] = result_copy['full_response'][:100] + '...' if len(result_copy['full_response']) > 100 else result_copy['full_response']
                
                all_results.append(result_copy)
        
        # Create DataFrame and select columns to include
        df = pd.DataFrame(all_results)
        
        # Make sure these columns come first in the CSV
        first_columns = ['model', 'question_id', 'question', 'attempt', 'model_grade', 
                         'extracted_grade', 'accuracy', 'consistency_score', 'grade_stability',
                         'response_time', 'confidence']
        
        # Get available columns from the ones we want first
        available_first = [col for col in first_columns if col in df.columns]
        
        # Get remaining columns (except very long text fields to keep CSV manageable)
        exclude_from_csv = ['full_response', 'prompt', 'model_answer', 'student_answer']
        other_columns = [col for col in df.columns if col not in available_first and col not in exclude_from_csv]
        
        # Reorder columns
        df = df[available_first + other_columns]
        
        # Write to CSV
        df.to_csv(detailed_path, index=False)
        
        # Create a DataFrame with just the core info and full responses
        response_df = pd.DataFrame([{
            'model': r['model'], 
            'question_id': r['question_id'],
            'attempt': r.get('attempt', 1),
            'prompt': r.get('prompt', ''),
            'full_response': r.get('full_response', '')
        } for r in all_results])
        
        response_df.to_csv(responses_path, index=False)

# Removed _calculate_summary method as it was only used for Excel export
