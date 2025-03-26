"""
Setup script for the Ollama Grader Evaluator package.
"""

from setuptools import setup, find_packages

setup(
    name="ollama_grader_evaluator",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "requests",
    ],
    entry_points={
        "console_scripts": [
            "ollama-grader=ollama_grader_evaluator.app:main",
        ],
    },
    author="",
    author_email="",
    description="A tool for evaluating LLM grading capabilities across different Ollama models",
    keywords="ollama, evaluation, grading",
    python_requires=">=3.6",
)
