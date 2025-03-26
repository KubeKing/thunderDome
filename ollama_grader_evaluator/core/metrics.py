"""
Metrics calculation for the Ollama Grader Evaluator.
"""

import re
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from ollama_grader_evaluator.utils.helpers import calculate_text_similarity


def extract_grade(response: str) -> Tuple[float, str]:
    """
    Extract a grade from the model's response.
    
    Args:
        response: Text response from the model
        
    Returns:
        Tuple of (extracted_grade, confidence_level)
    """
    # Try to find a grade in format "Grade: X.X" or similar
    grade_pattern = r"(?:grade|score|rating|mark):\s*([0-9]\.[0-9]|[01])"
    match = re.search(grade_pattern, response.lower())
    
    if match:
        return float(match.group(1)), "high"
    
    # Try to find a standalone decimal between 0 and 1
    decimal_pattern = r"(?<![a-zA-Z0-9])([0-9]\.[0-9]|[01])(?![0-9])"
    matches = re.findall(decimal_pattern, response)
    
    if matches:
        # If multiple matches, take the last one as it's likely the conclusion
        return float(matches[-1]), "medium"
    
    # Look for numbers written as words
    word_to_grade = {
        "zero": 0.0, "one": 1.0, "half": 0.5,
        "zero point five": 0.5, "point five": 0.5,
        "0": 0.0, "1": 1.0, "0.5": 0.5
    }
    
    for word, grade in word_to_grade.items():
        if word in response.lower():
            return grade, "low"
    
    # Default fallback
    return 0.5, "very low"


def calculate_consistency(attempts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate consistency metrics across multiple attempts.
    
    Args:
        attempts: List of attempt data dictionaries
        
    Returns:
        Dictionary of consistency metrics
    """
    if not attempts:
        return {
            "grade_stability": 0.0,
            "response_similarity": 0.0,
            "consistency_score": 0.0
        }
        
    # Extract grades from attempts
    grades = [a["extracted_grade"] for a in attempts]
    
    # Calculate grade stability - inverse of standard deviation (scaled)
    grade_std = np.std(grades)
    # Scale between 0 and 1 where 0 = completely inconsistent, 1 = perfect consistency
    grade_stability = max(0.0, min(1.0, 1.0 - (grade_std * 2)))
    
    # Calculate response similarity using text comparison
    responses = [a["full_response"] for a in attempts]
    similarity_scores = []
    
    # Compare each response with each other
    for i in range(len(responses)):
        for j in range(i+1, len(responses)):
            similarity = calculate_text_similarity(responses[i], responses[j])
            similarity_scores.append(similarity)
    
    # Average similarity
    response_similarity = np.mean(similarity_scores) if similarity_scores else 0.0
    
    # Combined consistency score (weighted average)
    consistency_score = (grade_stability * 0.7) + (response_similarity * 0.3)
    
    return {
        "grade_stability": grade_stability,
        "response_similarity": response_similarity,
        "consistency_score": consistency_score,
        "grades": grades,
        "grade_std": grade_std
    }


def calculate_accuracy(extracted_grade: float, model_grade: float) -> float:
    """
    Calculate the accuracy of the extracted grade against the model grade.
    
    Args:
        extracted_grade: Grade extracted from the model's response
        model_grade: Expected grade from the reference data
        
    Returns:
        Accuracy score between 0.0 and 1.0
    """
    return 1.0 - abs(extracted_grade - model_grade)


def create_prompt(question: str, model_answer: str, student_answer: str) -> str:
    """
    Create a prompt for the model to grade a student's answer.
    
    Args:
        question: The question being asked
        model_answer: The correct answer
        student_answer: The student's answer to grade
        
    Returns:
        Formatted prompt string
    """
    prompt = f"""Question: {question}

Correct Answer: {model_answer}

Student's Answer: {student_answer}

Grade the student's answer based on the correct answer from (0.0 - 1.0)"""
    return prompt
