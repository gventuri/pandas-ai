""" Base class to implement a new LLM. """

import ast
import re
from typing import Optional

import astor

from ..constants import END_CODE_TAG, START_CODE_TAG
from ..exceptions import (
    APIKeyNotFoundError,
    MethodNotImplementedError,
    NoCodeFoundError,
)


class LLM:
    """Base class to implement a new LLM."""

    last_prompt: Optional[str] = None

    @property
    def type(self) -> str:
        """
        Return type of LLM.

        Raises:
            APIKeyNotFoundError: Type has not been implemented

        Returns:
            str: Type of LLM a string
        """
        raise APIKeyNotFoundError("Type has not been implemented")

    def _remove_imports(self, code: str) -> str:
        tree = ast.parse(code)
        new_body = [
            node
            for node in tree.body
            if not isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        new_tree = ast.Module(body=new_body)
        return astor.to_source(new_tree)

    def _polish_code(self, code: str) -> str:
        """
        Polish the code by removing the leading "python" or "py",  \
        removing the imports and removing trailing spaces and new lines.

        Args:
            code (str): Code

        Returns:
            str: Polished code
        """
        if re.match(r"^(python|py)", code):
            code = re.sub(r"^(python|py)", "", code)
        if re.match(r"^`.*`$", code):
            code = re.sub(r"^`(.*)`$", r"\1", code)
        self._remove_imports(code)
        code = code.strip()
        return code

    def _is_python_code(self, string):
        try:
            ast.parse(string)
            return True
        except SyntaxError:
            return False

    def _extract_code(self, response: str, separator: str = "```") -> str:
        """
        Extract the code from the response.

        Args:
            response (str): Response
            separator (str, optional): Separator. Defaults to "```".

        Raises:
            NoCodeFoundError: No code found in the response

        Returns:
            str: Extracted code from the response
        """
        code = response
        if len(response.split(separator)) > 1:
            code = response.split(separator)[1]
        match = re.search(rf"{START_CODE_TAG}(.*){END_CODE_TAG}", code, re.DOTALL)
        if match:
            code = match.group(1).strip()
        if not self._is_python_code(code):
            raise NoCodeFoundError("No code found in the response")
        code = self._polish_code(code)
        return code

    def call(self, instruction: str, value: str) -> None:
        """
        Execute the LLM with given prompt.

        Args:
            instruction (str): Prompt
            value (str): Value

        Raises:
            MethodNotImplementedError: Call method has not been implemented
        """
        raise MethodNotImplementedError("Call method has not been implemented")

    def generate_code(self, instruction: str, prompt: str) -> str:
        """
        Generate the code based on the instruction and the given prompt.

        Returns:
            str: Code
        """
        return self._extract_code(self.call(instruction, prompt))
