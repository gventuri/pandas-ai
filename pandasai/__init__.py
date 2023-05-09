""" PandasAI is a wrapper around a LLM to make dataframes convesational """
import ast
import io
from contextlib import redirect_stdout
from datetime import date
from typing import Optional

import astor
import pandas as pd

from .constants import (
    END_CODE_TAG,
    START_CODE_TAG,
    WHITELISTED_BUILTINS,
    WHITELISTED_LIBRARIES,
)
from .exceptions import LLMNotFoundError
from .helpers.anonymizer import anonymize_dataframe_head
from .helpers.notebook import Notebook
from .llm.base import LLM


# pylint: disable=too-many-instance-attributes disable=too-many-arguments
class PandasAI:
    """PandasAI is a wrapper around a LLM to make dataframes conversational"""

    _task_instruction: str = """
Today is {today_date}.
You are provided with a pandas dataframe (df) with {num_rows} rows and {num_columns} columns.
This is the result of `print(df.head({rows_to_display}))`:
{df_head}.

Return the python code (do not import anything) and make sure to prefix the requested python code with {START_CODE_TAG} exactly and suffix the code with {END_CODE_TAG} exactly to get the answer to the following question:
"""
    _response_instruction: str = """
Question: {question}
Answer: {answer}

Rewrite the answer to the question in a conversational way.
"""

    _error_correct_instruction: str = """
Today is {today_date}.
You are provided with a pandas dataframe (df) with {num_rows} rows and {num_columns} columns.
This is the result of `print(df.head({rows_to_display}))`:
{df_head}.

The user asked the following question:
{question}

You generated this python code:
{code}

It fails with the following error:
{error_returned}

Correct the python code and return a new python code (do not import anything) that fixes the above mentioned error. Do not generate the same code again.
Make sure to prefix the requested python code with {START_CODE_TAG} exactly and suffix the code with {END_CODE_TAG} exactly.
    """
    _llm: LLM
    _verbose: bool = False
    _is_conversational_answer: bool = True
    _enforce_privacy: bool = False
    _max_retries: int = 3
    _is_notebook: bool = False
    _original_instructions: dict = {
        "question": None,
        "df_head": None,
        "num_rows": None,
        "num_columns": None,
        "rows_to_display": None,
    }
    last_code_generated: Optional[str] = None
    code_output: Optional[str] = None

    def __init__(
            self,
            llm=None,
            conversational=True,
            verbose=False,
            enforce_privacy=False,
    ):
        if llm is None:
            raise LLMNotFoundError(
                "An LLM should be provided to instantiate a PandasAI instance"
            )
        self._llm = llm
        self._is_conversational_answer = conversational
        self._verbose = verbose
        self._enforce_privacy = enforce_privacy

        self.notebook = Notebook()
        self._in_notebook = self.notebook.in_notebook()

    def conversational_answer(self, question: str, code: str, answer: str) -> str:
        """Return the conversational answer"""
        if self._enforce_privacy:
            # we don't want to send potentially sensitive data to the LLM server
            # if the user has set enforce_privacy to True
            return answer

        instruction = self._response_instruction.format(
            question=question, code=code, answer=answer
        )
        return self._llm.call(instruction, "")

    def run(
            self,
            data_frame: pd.DataFrame,
            prompt: str,
            is_conversational_answer: bool = None,
            show_code: bool = False,
            anonymize_df: bool = True,
            use_error_correction_framework: bool = True,
    ) -> str:
        """Run the LLM with the given prompt"""
        self.log(f"Running PandasAI with {self._llm.type} LLM...")

        rows_to_display = 0 if self._enforce_privacy else 5

        df_head = data_frame.head(rows_to_display)
        if anonymize_df:
            df_head = anonymize_dataframe_head(df_head)

        code = self._llm.generate_code(
            self._task_instruction.format(
                today_date=date.today(),
                df_head=df_head,
                num_rows=data_frame.shape[0],
                num_columns=data_frame.shape[1],
                rows_to_display=rows_to_display,
                START_CODE_TAG=START_CODE_TAG,
                END_CODE_TAG=END_CODE_TAG,
            ),
            prompt,
        )
        self._original_instructions = {
            "question": prompt,
            "df_head": df_head,
            "num_rows": data_frame.shape[0],
            "num_columns": data_frame.shape[1],
            "rows_to_display": rows_to_display,
        }
        self.last_code_generated = code
        self.log(
            f"""
Code generated:
```
{code}
```"""
        )
        if show_code and self._in_notebook:
            self.notebook.create_new_cell(code)

        answer = self.run_code(
            code,
            data_frame,
            use_error_correction_framework=use_error_correction_framework,
        )
        self.code_output = answer
        self.log(f"Answer: {answer}")

        if is_conversational_answer is None:
            is_conversational_answer = self._is_conversational_answer
        if is_conversational_answer:
            answer = self.conversational_answer(prompt, code, answer)
            self.log(f"Conversational answer: {answer}")
        return answer

    def remove_unsafe_imports(self, code: str) -> str:
        """Remove non-whitelisted imports from the code to prevent malicious code execution"""

        tree = ast.parse(code)
        new_body = [
            node
            for node in tree.body
            if not (
                    isinstance(node, (ast.Import, ast.ImportFrom))
                    and any(alias.name not in WHITELISTED_LIBRARIES for alias in node.names)
            )
        ]
        new_tree = ast.Module(body=new_body)
        return astor.to_source(new_tree).strip()

    def limit_context_dict(self, data_frame: pd.DataFrame):
        """create the dictionary that has context about variables and allowed builtins"""
        return {
            "pd": pd,
            "df": data_frame,
            "__builtins__": {
                "pd": pd,
                "df": data_frame,
                **{
                    builtin: __builtins__[builtin]
                    for builtin in WHITELISTED_BUILTINS
                },
            },
        }

    def run_code(
            self,
            code: str,
            data_frame: pd.DataFrame,
            use_error_correction_framework: bool = True,
    ) -> str:
        # pylint: disable=W0122 disable=W0123 disable=W0702:bare-except
        """Run the code in the current context and return the result"""

        # Redirect standard output to a StringIO buffer
        with redirect_stdout(io.StringIO()) as output:
            # Execute the code
            count = 0
            code_to_run = self.remove_unsafe_imports(code)
            while count < self._max_retries:
                try:
                    exec(code_to_run, self.limit_context_dict(data_frame))
                    code = code_to_run
                    break
                except Exception as e:  # pylint: disable=W0718 disable=C0103
                    if not use_error_correction_framework:
                        raise e

                    count += 1
                    error_correcting_instruction = (
                        self._error_correct_instruction.format(
                            today_date=date.today(),
                            code=code,
                            error_returned=e,
                            START_CODE_TAG=START_CODE_TAG,
                            END_CODE_TAG=END_CODE_TAG,
                            question=self._original_instructions["question"],
                            df_head=self._original_instructions["df_head"],
                            num_rows=self._original_instructions["num_rows"],
                            num_columns=self._original_instructions["num_columns"],
                            rows_to_display=self._original_instructions[
                                "rows_to_display"
                            ],
                        )
                    )
                    code_to_run = self._llm.generate_code(
                        error_correcting_instruction, ""
                    )

        captured_output = output.getvalue()

        # Evaluate the last line and return its value or the captured output
        lines = code.strip().split("\n")
        last_line = lines[-1].strip()
        if last_line.startswith("print(") and last_line.endswith(")"):
            last_line = last_line[6:-1]
        try:
            return eval(last_line,self.limit_context_dict(data_frame))
        except Exception:  # pylint: disable=W0718
            return captured_output

    def log(self, message: str):
        """Log a message"""
        if self._verbose:
            print(message)
