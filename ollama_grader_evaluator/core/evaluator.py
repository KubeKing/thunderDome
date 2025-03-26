"""
Evaluation logic for the Ollama Grader Evaluator.
"""

import time
import threading
from typing import Dict, List, Any, Callable, Optional
import pandas as pd

from ollama_grader_evaluator.core.ollama_client import OllamaClient
from ollama_grader_evaluator.core.metrics import extract_grade, calculate_accuracy, calculate_consistency, create_prompt
from ollama_grader_evaluator.utils.helpers import format_time_remaining


class Evaluator:
    """
    Handles the evaluation of models against grading tasks.
    """
    
    def __init__(self, update_status_callback: Optional[Callable[[str, Optional[str]], None]] = None):
        """
        Initialize the evaluator.
        
        Args:
            update_status_callback: Callback function for status updates
        """
        self.ollama_client = OllamaClient()
        self.update_status_callback = update_status_callback
        self.results = {}
        self.current_progress = 0
        self.total_tasks = 0
        self.evaluation_start_time = None
        self.should_stop = False
        
        # Number of attempts per question
        self.attempts_per_question = 5
    
    def evaluate(self, data: pd.DataFrame, selected_models: List[str], 
                on_progress: Callable[[float, str, Optional[str]], None], 
                on_complete: Callable[[], None]) -> None:
        """
        Start the evaluation process in a separate thread.
        
        Args:
            data: DataFrame containing the questions and answers
            selected_models: List of model names to evaluate
            on_progress: Callback for progress updates
            on_complete: Callback for evaluation completion
        """
        # Reset state
        self.current_progress = 0
        self.should_stop = False
        self.results = {model: [] for model in selected_models}
        
        # Calculate total tasks
        self.total_tasks = len(data) * len(selected_models) * self.attempts_per_question
        self._update_status(f"Total tasks: {self.total_tasks} "
                          f"({len(data)} questions × {len(selected_models)} models × "
                          f"{self.attempts_per_question} attempts)", "important")
        
        # Record start time for time remaining calculations
        self.evaluation_start_time = time.time()
        
        # Define the evaluation function that will be called in the thread
        def evaluation_thread_func():
            self._run_evaluation(data, selected_models, on_progress)
            on_complete()
        
        # Start evaluation in a separate thread
        threading.Thread(target=evaluation_thread_func, daemon=True).start()
    
    def _run_evaluation(self, data: pd.DataFrame, selected_models: List[str], 
                       on_progress: Callable[[float, str, Optional[str]], None]) -> None:
        """
        Run the evaluation process.
        
        Args:
            data: DataFrame containing the questions and answers
            selected_models: List of model names to evaluate
            on_progress: Callback for progress updates
        """
        for model in selected_models:
            if self.should_stop:
                break
                
            # Warm up the model first
            self._warm_up_model(model)
            model_results = []  # Store all results for this model
            
            for i, row in data.iterrows():
                if self.should_stop:
                    break
                    
                question_id = i + 1
                
                self._update_status(f"Starting evaluation of Question {question_id} with {model}...", "important")
                on_progress(self.current_progress / self.total_tasks, 
                           f"Processing Q{question_id} with {model}...", 
                           self._estimate_time_remaining())
                
                # Store results for all attempts on this question
                question_attempts = []
                prompt = create_prompt(row["Question"], row["Model Answer"], row["Student Answer"])
                model_grade = float(row["Model Grade"])
                
                # Run multiple attempts for each question
                for attempt in range(1, self.attempts_per_question + 1):
                    if self.should_stop:
                        break
                        
                    status_msg = f"Q{question_id}, Attempt #{attempt}/{self.attempts_per_question} with {model}"
                    on_progress(self.current_progress / self.total_tasks, status_msg, self._estimate_time_remaining())
                    self._update_status(status_msg)
                    
                    try:
                        # Query the model
                        response = self.ollama_client.query(model, prompt)
                        
                        full_response = response["text"]
                        response_time = response["response_time"]
                        
                        # Parse grade from response
                        extracted_grade, confidence = extract_grade(full_response)
                        
                        # Calculate accuracy
                        accuracy = calculate_accuracy(extracted_grade, model_grade)
                        
                        # Store this attempt
                        attempt_data = {
                            "question_id": question_id,
                            "question": row["Question"],
                            "student_answer": row["Student Answer"],
                            "model_answer": row["Model Answer"],
                            "model_grade": model_grade,
                            "attempt": attempt,
                            "extracted_grade": extracted_grade,
                            "accuracy": accuracy,
                            "confidence": confidence,
                            "response_time": response_time,
                            "full_response": full_response,
                            "prompt": prompt
                        }
                        
                        question_attempts.append(attempt_data)
                        
                    except Exception as e:
                        error_msg = f"Error processing Q{question_id}, Attempt #{attempt} with {model}: {str(e)}"
                        self._update_status(error_msg, "error")
                        
                        # Add error entry
                        question_attempts.append({
                            "question_id": question_id,
                            "attempt": attempt,
                            "error": str(e),
                            "accuracy": 0.0,
                            "confidence": "very low",
                            "response_time": 0.0
                        })
                    
                    # Update progress
                    self.current_progress += 1
                    progress_percentage = self.current_progress / self.total_tasks
                    on_progress(progress_percentage, status_msg, self._estimate_time_remaining())
                
                # Calculate consistency metrics for this question
                consistency_metrics = calculate_consistency(question_attempts)
                
                # Add consistency metrics to each attempt
                for attempt in question_attempts:
                    attempt.update(consistency_metrics)
                    model_results.append(attempt)
                
                # Print question summary
                self._print_question_summary(model, question_id, question_attempts, consistency_metrics)
            
            # Store all results for this model
            self.results[model] = model_results
            
            # Print model summary
            self._print_model_summary(model, model_results)
    
    def stop_evaluation(self) -> None:
        """Stop any running evaluation."""
        self.should_stop = True
    
    def get_results(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get the evaluation results.
        
        Returns:
            Dictionary of results, keyed by model name
        """
        return self.results
    
    def _warm_up_model(self, model: str) -> bool:
        """
        Send a simple query to load the model into GPU memory.
        
        Args:
            model: Name of the model to warm up
            
        Returns:
            True if the model was loaded successfully, False otherwise
        """
        self._update_status(f"Loading model {model} to GPU...", "important")
        result = self.ollama_client.warm_up_model(model)
        
        if result:
            self._update_status(f"Model {model} loaded successfully", "success")
        else:
            self._update_status(f"Error loading model {model}", "error")
            
        return result
    
    def _estimate_time_remaining(self) -> Optional[str]:
        """
        Estimate the remaining time based on progress.
        
        Returns:
            Formatted time remaining string, or None if not enough data
        """
        if self.evaluation_start_time is None or self.current_progress == 0:
            return None
            
        elapsed_time = time.time() - self.evaluation_start_time
        if elapsed_time <= 0:
            return None
            
        # Estimate total time based on progress so far
        estimated_total_time = elapsed_time * (self.total_tasks / self.current_progress)
        remaining_time = estimated_total_time - elapsed_time
        
        return format_time_remaining(remaining_time)
    
    def _update_status(self, message: str, tag: Optional[str] = None) -> None:
        """
        Update the status with a new message.
        
        Args:
            message: Status message
            tag: Optional tag for the message
        """
        if self.update_status_callback is not None:
            self.update_status_callback(message, tag)
        else:
            # Fallback to printing if no callback is provided
            print(f"[{time.strftime('%H:%M:%S')}] {message}")
    
    def _print_question_summary(self, model: str, question_id: int, 
                               attempts: List[Dict[str, Any]], 
                               consistency: Dict[str, Any]) -> None:
        """
        Print summary statistics for a question to the terminal.
        
        Args:
            model: Name of the model
            question_id: ID of the question
            attempts: List of attempt data dictionaries
            consistency: Dictionary of consistency metrics
        """
        print(f"\n----- Model: {model}, Question {question_id} -----")
        print(f"Attempts: {len(attempts)}")
        print(f"Grades: {[round(a['extracted_grade'], 2) for a in attempts]}")
        print(f"Grade Stability: {consistency['grade_stability']:.3f}")
        print(f"Response Similarity: {consistency['response_similarity']:.3f}")
        print(f"Overall Consistency: {consistency['consistency_score']:.3f}")
        print(f"Average Accuracy: {sum([a['accuracy'] for a in attempts]) / len(attempts):.3f}")
        print(f"Average Response Time: {sum([a['response_time'] for a in attempts]) / len(attempts):.2f}s")
    
    def _print_model_summary(self, model: str, results: List[Dict[str, Any]]) -> None:
        """
        Print overall summary statistics for a model to the terminal.
        
        Args:
            model: Name of the model
            results: List of result data dictionaries
        """
        print(f"\n{'='*20} {model} SUMMARY {'='*20}")
        
        # Group results by question
        questions = {}
        for r in results:
            q_id = r["question_id"]
            if q_id not in questions:
                questions[q_id] = []
            questions[q_id].append(r)
        
        # Calculate overall metrics
        all_accuracies = [r["accuracy"] for r in results]
        all_consistencies = [r["consistency_score"] for r in results if "consistency_score" in r]
        all_times = [r["response_time"] for r in results]
        
        print(f"Questions evaluated: {len(questions)}")
        print(f"Total attempts: {len(results)}")
        
        if all_accuracies:
            print(f"Overall Accuracy: {sum(all_accuracies) / len(all_accuracies):.3f}")
        
        if all_consistencies:
            print(f"Overall Consistency: {sum(all_consistencies) / len(all_consistencies):.3f}")
        
        if all_times:
            print(f"Average Response Time: {sum(all_times) / len(all_times):.2f}s")
            
        print("="*60)
