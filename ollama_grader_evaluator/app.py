"""
Main entry point for the Ollama Grader Evaluator application.
"""

import tkinter as tk
from ollama_grader_evaluator.ui.main_window import MainWindow


def main():
    """Initialize and run the application."""
    root = tk.Tk()
    app = MainWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
