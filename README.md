# Ollama Grader Evaluator

A Python application that evaluates how well different Ollama models grade short-answer computer science questions.

## Overview

This tool allows you to:
- Upload a CSV or Excel file containing questions, model answers, student answers, and reference grades
- Select which Ollama models to evaluate
- Compare how accurately different models grade student answers
- Analyze performance metrics like grading accuracy, consistency, and response time
- Export results in various formats for further analysis

## Requirements

- Python 3.8+
- Ollama running locally on port 11434
- Required Python packages:
  - tkinter
  - pandas
  - numpy
  - matplotlib
  - requests
  - openpyxl (for Excel support)

## Installation

1. Ensure Ollama is installed and running on your system.
2. Install required Python dependencies:

```
pip install pandas numpy matplotlib requests openpyxl
```

## Usage

1. Run the application:

```
python ollama_grader_evaluator.py
```

2. Click "Browse" to select your input CSV or Excel file.
3. Select the models you want to evaluate.
4. Click "Run Evaluation" to start the process.
5. View results and export them using the "Export Results" button.

## Input Data Format

The input file should be a CSV or Excel file with the following columns:
- **Question**: The computer science question
- **Model Answer**: The correct answer (reference)
- **Student Answer**: The student's response to be graded
- **Model Grade**: The reference grade (0.0-1.0) for evaluation

Example format:
```
Question,Model Answer,Student Answer,Model Grade
"What is a binary search?","Binary search is...",Student response...",0.8
```

## Understanding Results

The evaluation provides several metrics:
- **Average Accuracy**: How close the model's grade is to the reference grade (higher is better)
- **Consistency**: How consistently the model grades across different questions (higher is better)
- **Average Response Time**: How long the model takes to generate a grade
- **Extraction Confidence**: How confidently the grade could be extracted from the model's response

## Supported Models

The application is pre-configured with the following models:
- gemma3:12b
- qwen2.5:14b
- exaone-deep:7.8b
- deepseek-r1:14b
- deepseek-r1:8b
- mistral:latest

Ensure these models are available in your Ollama installation or modify the code to include your available models.
