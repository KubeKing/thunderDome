# Ollama Grader Evaluator

A tool for evaluating LLM grading capabilities across different Ollama models.

## Overview

This application helps evaluate how consistently and accurately different Ollama language models grade student answers. It provides visualization tools, metrics for consistency and accuracy, and export options.

## Project Structure

The project has been modularized into the following structure:

```
ollama_grader_evaluator/
├── __init__.py                # Package initialization
├── app.py                     # Main application entry point
├── ui/                        # UI components
│   ├── __init__.py
│   ├── main_window.py         # Main application window
│   ├── file_input.py          # File selection components
│   ├── model_selection.py     # Model selection checkboxes
│   ├── progress_widgets.py    # Progress bar and status display
│   └── results_display.py     # Results visualization and tables
├── core/                      # Core functionality
│   ├── __init__.py
│   ├── data_manager.py        # Data loading and processing
│   ├── ollama_client.py       # Ollama API interaction
│   ├── evaluator.py           # Evaluation logic
│   ├── metrics.py             # Metrics calculation (consistency, accuracy)
│   └── export.py              # Export functionality
└── utils/                     # Utility functions
    ├── __init__.py
    └── helpers.py             # Common helper functions
```

## Prerequisites

- Python 3.6+
- Ollama running locally (with API accessible at http://localhost:11434)
- Tkinter (usually comes with Python)

## Installation

### From Source

1. Clone the repository:
   ```
   git clone [repository_url]
   cd ollama-grader-evaluator
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python run_evaluator.py
   ```

### Using pip

```
pip install .
ollama-grader
```

## Input Data Format

The application expects a CSV or Excel file with the following columns:

- `Question`: The question being asked
- `Model Answer`: The correct/reference answer
- `Student Answer`: The student's answer to be graded
- `Model Grade`: The expected grade (between 0.0 and 1.0)

## Features

- **Multi-model Evaluation**: Test multiple Ollama models simultaneously
- **Consistency Metrics**: Measure how consistently models grade the same answer
- **Accuracy Metrics**: Compare grades against expected/reference grades
- **Interactive UI**: Visual feedback on progress and results
- **Export Options**: Export results in various formats (CSV, Excel, JSON)

## How It Works

1. The application sends a carefully formatted grading prompt to each selected Ollama model
2. Each question is evaluated multiple times to measure consistency
3. Grades are extracted from the model's response using pattern matching
4. Results are analyzed and presented with visualizations

## License

[License information]

## Acknowledgments

[Any acknowledgments]
