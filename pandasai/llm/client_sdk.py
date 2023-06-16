from llm_client import BaseLLMClient

from pandasai import LLM
from pandasai.prompts.base import Prompt


class ClientSDK(LLM):
    def __init__(self, client: BaseLLMClient, **kwargs):
        self._client = client
        self._kwargs = kwargs

    @property
    def type(self) -> str:
        return "generic-llm-client"

    def call(self, instruction: Prompt, value: str, suffix: str = "") -> str:
        return self._client.text_completion(
            str(instruction) + value + suffix, **self._kwargs
        )[0]
