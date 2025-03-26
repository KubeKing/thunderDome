"""
Helper functions for the Ollama Grader Evaluator.
"""

import os
import time
from typing import Dict, List, Any, Tuple, Optional
import difflib


def format_time_remaining(seconds: float) -> str:
    """
    Format a time in seconds into a human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string (e.g., "5 seconds", "2 minutes", "1h 30m")
    """
    if seconds < 60:
        return f"{int(seconds)} seconds"
    elif seconds < 3600:
        return f"{int(seconds / 60)} minutes"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}h {minutes}m"


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate the similarity between two text strings using difflib.
    
    Args:
        text1: First text string
        text2: Second text string
        
    Returns:
        Similarity ratio between 0.0 and 1.0
    """
    seq = difflib.SequenceMatcher(None, text1, text2)
    return seq.ratio()


def generate_timestamp_filename(base_name: str, extension: str) -> str:
    """
    Generate a filename with a timestamp.
    
    Args:
        base_name: Base filename
        extension: File extension (without the dot)
        
    Returns:
        Filename with timestamp (e.g., "results_20250324_124500.csv")
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}.{extension}"
