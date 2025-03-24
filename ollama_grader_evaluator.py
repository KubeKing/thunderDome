import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
import threading
import json
import re
import os
import time
import requests
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from io import StringIO
import difflib
import sys

class OllamaGraderEvaluator:
    def __init__(self, root):
        self.root = root
        self.root.title("Ollama Grader Evaluator")
        self.root.geometry("1000x800")
        self.root.minsize(800, 600)
        
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
        
        self.selected_models = {model: tk.BooleanVar(value=True) for model in self.models}
        self.data = None
        self.results = {}
        self.current_progress = 0
        self.total_tasks = 0
        self.evaluation_start_time = None
        
        # Number of attempts per question
        self.attempts_per_question = 5
        
        self.create_widgets()
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # File input section
        file_frame = ttk.LabelFrame(main_frame, text="Input Data", padding=10)
        file_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(file_frame, text="Select CSV or Excel file:").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        self.file_path_var = tk.StringVar()
        file_path_entry = ttk.Entry(file_frame, textvariable=self.file_path_var, width=50)
        file_path_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        browse_button = ttk.Button(file_frame, text="Browse", command=self.browse_file)
        browse_button.grid(row=0, column=2, padx=5, pady=5)
        
        # Model selection section
        model_frame = ttk.LabelFrame(main_frame, text="Select Models to Evaluate", padding=10)
        model_frame.pack(fill=tk.X, pady=5)
        
        for i, model in enumerate(self.models):
            row, col = divmod(i, 3)
            cb = ttk.Checkbutton(
                model_frame, 
                text=model, 
                variable=self.selected_models[model],
                onvalue=True,
                offvalue=False
            )
            cb.grid(row=row, column=col, sticky=tk.W, padx=10, pady=5)
        
        # Progress section
        progress_frame = ttk.LabelFrame(main_frame, text="Evaluation Progress", padding=10)
        progress_frame.pack(fill=tk.X, pady=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            progress_frame, 
            orient=tk.HORIZONTAL, 
            length=500, 
            mode='determinate',
            variable=self.progress_var
        )
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        # Create a frame for status and time estimate
        status_estimate_frame = ttk.Frame(progress_frame)
        status_estimate_frame.pack(fill=tk.X, pady=5)
        
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(status_estimate_frame, textvariable=self.status_var)
        status_label.pack(side=tk.LEFT, pady=5)
        
        # Add time remaining estimate
        self.time_left_var = tk.StringVar(value="")
        time_left_label = ttk.Label(status_estimate_frame, textvariable=self.time_left_var)
        time_left_label.pack(side=tk.RIGHT, pady=5)
        
        # Add detailed status text widget
        status_frame = ttk.Frame(progress_frame)
        status_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(status_frame, text="Detailed Status:").pack(anchor=tk.W)
        
        # Use scrolledtext instead of regular Text widget for better usability
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
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.run_button = ttk.Button(
            button_frame, 
            text="Run Evaluation", 
            command=self.run_evaluation,
            state=tk.DISABLED
        )
        self.run_button.pack(side=tk.RIGHT, padx=5)
        
        # Results frame (initially empty)
        self.results_frame = ttk.LabelFrame(main_frame, text="Results", padding=10)
        self.results_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Canvas for matplotlib
        self.canvas_frame = ttk.Frame(self.results_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
    def update_detailed_status(self, message, tag=None):
        timestamp = time.strftime("%H:%M:%S")
        self.detailed_status.insert(tk.END, f"[{timestamp}] {message}\n", tag if tag else "")
        self.detailed_status.see(tk.END)
        self.detailed_status.update_idletasks()  # Force update
        
        # Also print to console for logging
        print(f"[{timestamp}] {message}")
    
    def browse_file(self):
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
            self.load_data(filename)
    
    def load_data(self, file_path):
        try:
            if file_path.endswith(('.xlsx', '.xls')):
                self.data = pd.read_excel(file_path)
            else:
                self.data = pd.read_csv(file_path)
                
            # Validate that the required columns exist
            required_columns = ["Question", "Model Answer", "Student Answer", "Model Grade"]
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            
            if missing_columns:
                messagebox.showerror(
                    "Missing Columns", 
                    f"The following required columns are missing: {', '.join(missing_columns)}"
                )
                self.data = None
                self.run_button.configure(state=tk.DISABLED)
            else:
                self.status_var.set(f"Loaded {len(self.data)} questions")
                self.run_button.configure(state=tk.NORMAL)
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")
            self.data = None
            self.run_button.configure(state=tk.DISABLED)
    
    def run_evaluation(self):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded.")
            return
        
        selected_models = [model for model, var in self.selected_models.items() if var.get()]
        
        if not selected_models:
            messagebox.showerror("Error", "Please select at least one model.")
            return
        
        # Disable run button during evaluation
        self.run_button.configure(state=tk.DISABLED)
        
        # Clear detailed status
        self.detailed_status.delete(1.0, tk.END)
        self.update_detailed_status("Starting evaluation...", "important")
        
        # Reset progress
        self.progress_var.set(0)
        self.current_progress = 0
        
        # Record start time for time remaining calculations
        self.evaluation_start_time = time.time()
        self.time_left_var.set("Calculating...")
        
        # Each question is evaluated 5 times per model
        self.total_tasks = len(self.data) * len(selected_models) * self.attempts_per_question
        self.update_detailed_status(f"Total tasks: {self.total_tasks} ({len(self.data)} questions × {len(selected_models)} models × {self.attempts_per_question} attempts)")
        
        # Initialize results dictionary to hold grouped data by question/model
        # Use a better structure to track attempts
        self.results = {model: {} for model in selected_models}
        
        # Start evaluation in a separate thread
        threading.Thread(target=self.evaluation_thread, args=(selected_models,), daemon=True).start()
    
    def warm_up_model(self, model):
        """Send a simple query to load the model into GPU memory"""
        try:
            self.update_detailed_status(f"Loading model {model} to GPU...", "important")
            # Simple warm-up query
            response = self.query_ollama(model, "Hello, are you ready?")
            self.update_detailed_status(f"Model {model} loaded successfully", "success")
            return True
        except Exception as e:
            self.update_detailed_status(f"Error loading model {model}: {str(e)}", "error")
            return False

    def calculate_consistency(self, attempts):
        """Calculate consistency metrics across multiple attempts"""
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
                # Use difflib to calculate similarity
                seq = difflib.SequenceMatcher(None, responses[i], responses[j])
                similarity = seq.ratio()
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

    def print_question_summary(self, model, question_id, attempts, consistency):
        """Print summary statistics for a question to the terminal"""
        print(f"\n----- Model: {model}, Question {question_id} -----")
        print(f"Attempts: {len(attempts)}")
        print(f"Grades: {[round(a['extracted_grade'], 2) for a in attempts]}")
        print(f"Grade Stability: {consistency['grade_stability']:.3f}")
        print(f"Response Similarity: {consistency['response_similarity']:.3f}")
        print(f"Overall Consistency: {consistency['consistency_score']:.3f}")
        print(f"Average Accuracy: {np.mean([a['accuracy'] for a in attempts]):.3f}")
        print(f"Average Response Time: {np.mean([a['response_time'] for a in attempts]):.2f}s")

    def print_model_summary(self, model, results):
        """Print overall summary statistics for a model to the terminal"""
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
        print(f"Overall Accuracy: {np.mean(all_accuracies):.3f}")
        
        if all_consistencies:
            print(f"Overall Consistency: {np.mean(all_consistencies):.3f}")
        
        print(f"Average Response Time: {np.mean(all_times):.2f}s")
        print("="*60)

    def evaluation_thread(self, selected_models):
        self.status_var.set("Evaluating...")
        
        for model in selected_models:
            # Warm up the model first
            self.warm_up_model(model)
            model_results = []  # Store all results for this model
            
            for i, row in self.data.iterrows():
                question_id = i + 1
                question_key = f"Q{question_id}"
                
                self.update_detailed_status(f"Starting evaluation of Question {question_id} with {model}...", "important")
                self.status_var.set(f"Processing Q{question_id} with {model}...")
                
                # Store results for all attempts on this question
                question_attempts = []
                prompt = self.create_prompt(row)
                model_grade = float(row["Model Grade"])
                
                # Run multiple attempts for each question
                for attempt in range(1, self.attempts_per_question + 1):
                    try:
                        status_msg = f"Q{question_id}, Attempt #{attempt}/{self.attempts_per_question} with {model}"
                        self.status_var.set(status_msg)
                        self.update_detailed_status(status_msg)
                        
                        # Query the model
                        start_time = time.time()
                        response = self.query_ollama(model, prompt)
                        end_time = time.time()
                        response_time = end_time - start_time
                        
                        # Parse grade from response
                        extracted_grade, confidence = self.extract_grade(response)
                        
                        # Calculate accuracy
                        accuracy = 1.0 - abs(extracted_grade - model_grade)
                        
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
                            "full_response": response,
                            "prompt": prompt
                        }
                        
                        question_attempts.append(attempt_data)
                        
                    except Exception as e:
                        error_msg = f"Error processing Q{question_id}, Attempt #{attempt} with {model}: {str(e)}"
                        self.update_detailed_status(error_msg, "error")
                        
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
                    progress_percentage = (self.current_progress / self.total_tasks) * 100
                    self.progress_var.set(progress_percentage)
                    
                    # Calculate and update time remaining estimate
                    if self.evaluation_start_time is not None and self.current_progress > 0:
                        elapsed_time = time.time() - self.evaluation_start_time
                        if elapsed_time > 0:
                            # Estimate total time based on progress so far
                            estimated_total_time = elapsed_time * (self.total_tasks / self.current_progress)
                            remaining_time = estimated_total_time - elapsed_time
                            
                            # Format the time remaining
                            if remaining_time < 60:
                                time_str = f"Time left: {int(remaining_time)} seconds"
                            elif remaining_time < 3600:
                                time_str = f"Time left: {int(remaining_time / 60)} minutes"
                            else:
                                hours = int(remaining_time / 3600)
                                minutes = int((remaining_time % 3600) / 60)
                                time_str = f"Time left: {hours}h {minutes}m"
                            
                            self.time_left_var.set(time_str)
                
                # Calculate consistency metrics for this question
                consistency_metrics = self.calculate_consistency(question_attempts)
                
                # Add consistency metrics to each attempt
                for attempt in question_attempts:
                    attempt.update(consistency_metrics)
                    model_results.append(attempt)
                
                # Print question summary
                self.print_question_summary(model, question_id, question_attempts, consistency_metrics)
            
            # Store all results for this model
            self.results[model] = model_results
            
            # Print model summary
            self.print_model_summary(model, model_results)
                
        # Evaluation complete
        self.root.after(0, self.evaluation_complete)
    
    def create_prompt(self, row):
        prompt = f"""Question: {row['Question']}

Correct Answer: {row['Model Answer']}

Student's Answer: {row['Student Answer']}

Grade the student's answer based on the correct answer from (0.0 - 1.0)"""
        return prompt
    
    def query_ollama(self, model, prompt):
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                return f"Error: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    def extract_grade(self, response):
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
    
    def evaluation_complete(self):
        self.status_var.set("Evaluation complete!")
        self.run_button.configure(state=tk.NORMAL)
        
        # Clear previous results
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()
        
        # Display results
        self.display_results()
    
    def display_results(self):
        # Create figure for plotting
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))
        
        # Aggregate metrics by model
        model_metrics = {}
        
        for model, results in self.results.items():
            # Group results by question to calculate per-question metrics
            questions = {}
            for r in results:
                q_id = r["question_id"]
                if q_id not in questions:
                    questions[q_id] = []
                questions[q_id].append(r)
            
            # Calculate average accuracy across all attempts
            avg_accuracy = np.mean([r["accuracy"] for r in results])
            
            # Calculate average consistency score
            consistency_scores = [r.get("consistency_score", 0) for r in results if "consistency_score" in r]
            avg_consistency = np.mean(consistency_scores) if consistency_scores else 0
            
            # Calculate average response time
            avg_time = np.mean([r["response_time"] for r in results])
            
            model_metrics[model] = {
                "accuracy": avg_accuracy,
                "consistency": avg_consistency,
                "response_time": avg_time,
                "questions": len(questions)
            }
        
        models = list(model_metrics.keys())
        accuracies = [model_metrics[model]["accuracy"] for model in models]
        consistencies = [model_metrics[model]["consistency"] for model in models]
        times = [model_metrics[model]["response_time"] for model in models]
        
        # Plot 1: Average accuracy
        bars1 = ax1.bar(models, accuracies, color='skyblue')
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Average Accuracy')
        ax1.set_title('Average Grading Accuracy by Model')
        ax1.set_ylim(0, 1)
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
        ax3.set_xticklabels(models, rotation=45, ha='right')
        
        # Add value labels
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}s', ha='center', va='bottom', rotation=0)
        
        fig.tight_layout()
        
        # Add to canvas
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add detailed results table
        self.add_results_table()
        
        # Add export buttons
        export_frame = ttk.Frame(self.canvas_frame)
        export_frame.pack(pady=10)
        
        export_button = ttk.Button(
            export_frame, 
            text="Export Results", 
            command=self.export_results
        )
        export_button.pack(side=tk.LEFT, padx=5)
        
        export_csv_button = ttk.Button(
            export_frame, 
            text="Export as CSV", 
            command=self.export_as_csv
        )
        export_csv_button.pack(side=tk.LEFT, padx=5)
    
    def add_results_table(self):
        # Create frame for table
        table_frame = ttk.Frame(self.canvas_frame)
        table_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Calculate summary statistics
        summary_data = self.calculate_summary()
        
        # Create table headers
        columns = ["Model", "Avg Accuracy", "Consistency", "Avg Response Time", "Extraction Confidence"]
        
        for i, col in enumerate(columns):
            ttk.Label(table_frame, text=col, font=("TkDefaultFont", 10, "bold")).grid(
                row=0, column=i, padx=5, pady=5, sticky=tk.W
            )
        
        # Add data rows
        for i, model in enumerate(self.results.keys()):
            stats = summary_data[model]
            
            ttk.Label(table_frame, text=model).grid(
                row=i+1, column=0, padx=5, pady=2, sticky=tk.W
            )
            
            ttk.Label(table_frame, text=f"{stats['avg_accuracy']:.3f}").grid(
                row=i+1, column=1, padx=5, pady=2
            )
            
            ttk.Label(table_frame, text=f"{stats['consistency']:.3f}").grid(
                row=i+1, column=2, padx=5, pady=2
            )
            
            ttk.Label(table_frame, text=f"{stats['avg_time']:.2f}s").grid(
                row=i+1, column=3, padx=5, pady=2
            )
            
            ttk.Label(table_frame, text=f"{stats['confidence']}").grid(
                row=i+1, column=4, padx=5, pady=2
            )
    
    def calculate_summary(self):
        summary = {}
        
        for model, results in self.results.items():
            accuracies = [r.get("accuracy", 0) for r in results]
            times = [r.get("response_time", 0) for r in results]
            confidences = [r.get("confidence", "low") for r in results if "confidence" in r]
            
            # Count confidence levels
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
            most_common = max(confidence_counts.items(), key=lambda x: x[1])
            confidence_text = most_common[0]
            
            summary[model] = {
                "avg_accuracy": np.mean(accuracies),
                "consistency": 1.0 - np.std(accuracies),  # Higher is better
                "avg_time": np.mean(times),
                "confidence": confidence_text
            }
            
        return summary
    
    def export_results(self):
        try:
            # Get the directory where the script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Generate timestamp for unique filenames
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # Generate export paths for different formats
            excel_path = os.path.join(script_dir, f"evaluation_results_{timestamp}.xlsx")
            csv_path = os.path.join(script_dir, f"evaluation_results_{timestamp}.csv")
            json_path = os.path.join(script_dir, f"evaluation_results_{timestamp}.json")
            
            # Export in all formats
            self.export_to_excel(excel_path)
            self.export_to_csv(csv_path)
            self.export_to_json(json_path)
            
            # Notify user
            export_msg = f"Results exported to:\n- {excel_path}\n- {csv_path}\n- {json_path}"
            self.update_detailed_status(export_msg, "success")
            messagebox.showinfo("Export Complete", export_msg)
                
        except Exception as e:
            error_msg = f"Failed to export results: {str(e)}"
            self.update_detailed_status(error_msg, "error")
            messagebox.showerror("Export Error", error_msg)
    
    def export_to_excel(self, path):
        # Create a Pandas Excel writer
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = self.calculate_summary()
            summary_df = pd.DataFrame.from_dict(summary_data, orient='index')
            summary_df.reset_index(inplace=True)
            summary_df.rename(columns={'index': 'model'}, inplace=True)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Create sheets for each model
            for model, results in self.results.items():
                df = pd.DataFrame(results)
                
                # Select columns to export
                columns_to_keep = [
                    'question_id', 'question', 'model_grade', 
                    'extracted_grade', 'accuracy', 'confidence', 
                    'response_time'
                ]
                
                available_columns = [col for col in columns_to_keep if col in df.columns]
                df = df[available_columns]
                
                df.to_excel(writer, sheet_name=model.replace(":", "_"), index=False)
    
    def export_to_csv(self, path):
        # For CSV, we'll just export a summary of all models
        all_results = []
        for model, results in self.results.items():
            for result in results:
                result_copy = result.copy()
                result_copy['model'] = model
                all_results.append(result_copy)
        
        df = pd.DataFrame(all_results)
        df.to_csv(path, index=False)
    
    def export_to_json(self, path):
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def export_as_csv(self):
        """Export detailed results directly to CSV with full prompts and responses"""
        try:
            # Get the directory where the script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Generate timestamp for unique filenames
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # Set export paths
            export_path = os.path.join(script_dir, f"evaluation_detailed_{timestamp}.csv")
            response_path = os.path.join(script_dir, f"evaluation_full_responses_{timestamp}.csv")
            
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
            df.to_csv(export_path, index=False)
            
            # Create a DataFrame with just the core info and full responses
            response_df = pd.DataFrame([{
                'model': r['model'], 
                'question_id': r['question_id'],
                'attempt': r.get('attempt', 1),
                'prompt': r.get('prompt', ''),
                'full_response': r.get('full_response', '')
            } for r in all_results])
            
            response_df.to_csv(response_path, index=False)
            
            export_msg = f"Results exported to:\n- {export_path}\n- {response_path}"
            self.update_detailed_status(export_msg, "success")
            messagebox.showinfo("Export Complete", export_msg)
            
        except Exception as e:
            error_msg = f"Failed to export CSV: {str(e)}"
            self.update_detailed_status(error_msg, "error")
            messagebox.showerror("Export Error", error_msg)


def main():
    root = tk.Tk()
    app = OllamaGraderEvaluator(root)
    root.mainloop()


if __name__ == "__main__":
    main()
