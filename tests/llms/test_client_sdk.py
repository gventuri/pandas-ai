import pytest
from llm_client import OpenAIClient

from pandasai.llm.client_sdk import ClientSDK
from pandasai.prompts.base import Prompt
from unittest.mock import Mock, MagicMock


class TestClientSDK:
    """Unit tests for the llm-client wrapper LLM class"""

    @pytest.fixture
    def llm_client_mock(self):
        mock = MagicMock(OpenAIClient)
        mock.text_completion = Mock(return_value=["Custom response"])
        return mock

    @pytest.fixture
    def prompt(self):
        class MockPrompt(Prompt):
            text: str = "Hello"

        return MockPrompt()

    def test_llm_client_type(self, llm_client_mock):
        llm_client_wrapper = ClientSDK(llm_client_mock)

        assert llm_client_wrapper.type == "generic-llm-client"

    def test_llm_client_model_call(self, llm_client_mock, prompt):
        langchain_wrapper = ClientSDK(llm_client_mock)

        assert (
            langchain_wrapper.call(instruction=prompt, value="world", suffix="!")
            == "Custom response"
        )
