"""Cohere LLM"""

import os
from dotenv import load_dotenv
import cohere
from .base import LLM
from ..exceptions import APIKeyNotFoundError, UnsupportedCohereModelError

load_dotenv()


# pylint: disable=duplicate-code
class CohereLLM(LLM):
    """Cohere LLM"""

    _supported_chat_models = [
        "command",
        "command-nightly",
        "command-light",
        "command-light-nightly",
        "command-xlarge",
        "command-xlarge-nightly",
    ]

    api_token: str
    model: str = "command-xlarge-nightly"
    temperature: float = 0
    max_tokens: int = 512
    top_p: float = 1
    frequency_penalty: float = 0
    presence_penalty: float = 0.6
    stop: str = None

    def __init__(
        self,
        api_token: str = None,
        **kwargs,
    ):
        self.api_token = api_token or os.getenv("COHERE_API_KEY")
        if self.api_token is None:
            raise APIKeyNotFoundError("Cohere API key is required")
        cohere.api_key = self.api_token

        self._set_params(**kwargs)

    def _set_params(self, **kwargs):
        valid_params = [
            "model",
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop",
        ]
        for key, value in kwargs.items():
            if key in valid_params:
                setattr(self, key, value)

    def completion(self, prompt: str) -> str:
        """
        Query the completation API

        Args:
            prompt (str): Prompt

        Returns:
            str: LLM response
        """
        params = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }

        if self.stop is not None:
            params["stop"] = [self.stop]

        response = (cohere.Client(self.api_token)).generate(**params)

        return response.generations[0].text

    def call(self, instruction: str, value: str, suffix: str = "") -> str:
        """
        Call the Cohere LLM.

        Args:
            instruction (str): Instruction to pass
            value (str): Value to pass

        Raises:
            UnsupportedOpenAIModelError: Unsupported model

        Returns:
            str: Response
        """
        self.last_prompt = str(instruction) + str(value)

        if self.model in self._supported_chat_models:
            response = self.completion(str(instruction) + str(value))
        else:
            raise UnsupportedCohereModelError("Unsupported model")

        return response

    @property
    def type(self) -> str:
        return "cohere"
