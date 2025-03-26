"""
Client for interacting with the Ollama API.
"""

import time
import requests
from typing import Dict, Any, Optional


class OllamaClient:
    """
    Client for interacting with the Ollama API.
    """
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        Initialize the Ollama client.
        
        Args:
            base_url: Base URL for the Ollama API
        """
        self.base_url = base_url.rstrip('/')
        self.api_endpoint = f"{self.base_url}/api/generate"
        
    def query(self, model: str, prompt: str, timeout: int = 60) -> Dict[str, Any]:
        """
        Query the Ollama API with a prompt.
        
        Args:
            model: Name of the model to use
            prompt: Prompt to send to the model
            timeout: Request timeout in seconds
            
        Returns:
            Dictionary containing the model's response and metadata
            
        Raises:
            Exception: If the API request fails
        """
        try:
            start_time = time.time()
            
            response = requests.post(
                self.api_endpoint,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=timeout
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "text": result.get("response", ""),
                    "response_time": response_time,
                    "raw_response": result
                }
            else:
                error_message = f"Error: {response.status_code} - {response.text}"
                return {
                    "text": error_message,
                    "response_time": response_time,
                    "error": True
                }
                
        except Exception as e:
            return {
                "text": f"Error: {str(e)}",
                "response_time": 0.0,
                "error": True
            }
    
    def warm_up_model(self, model: str) -> bool:
        """
        Send a simple query to load the model into GPU memory.
        
        Args:
            model: Name of the model to warm up
            
        Returns:
            True if the model was loaded successfully, False otherwise
        """
        try:
            response = self.query(model, "Hello, are you ready?")
            return not response.get("error", False)
        except Exception:
            return False
