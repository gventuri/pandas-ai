"""TextGenWebui LLM"""

from typing import Optional

from .base import LLM
import requests


class TextGenWebuiLLM(LLM):
    """TextGenWebui LLM"""

    _output: str = 'print("Hello world")'
    api_url: str = (
        "http://localhost:5000/api/v1/generate"
    )

    def __init__(self, api_url: Optional[str] = None):
        if api_url is not None:
            self.api_url = api_url

    def query(self, payload):
        
        response = requests.post(
            self.api_url, json=payload, timeout=60
        )
        return response.json()


    def call(self, instruction: str, value: str) -> str:

        output = self.query(
            {"prompt": "<|prompter|>" + instruction + value + "<|endoftext|>"}
        )
        return output["results"][0]["text"]
    
    @property
    def type(self) -> str:
        return "text-gen-webui"
