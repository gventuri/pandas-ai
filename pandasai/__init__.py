# -*- coding: utf-8 -*-
"""
PandasAI is a wrapper around a LLM to make dataframes conversational

This module includes the implementation of basis  PandasAI class with methods to run
the LLMs models on Pandas dataframes. Following LLMs are implemented so far.

Example:

    This module is the Entry point of the `pandasai` package. Following is an example
    of how to use this Class.

    ```python
    import pandas as pd
    from pandasai import PandasAI

    # Sample DataFrame
    df = pd.DataFrame({
        "country": ["United States", "United Kingdom", "France", "Germany", "Italy",
        "Spain", "Canada", "Australia", "Japan", "China"],
        "gdp": [19294482071552, 2891615567872, 2411255037952, 3435817336832,
        1745433788416, 1181205135360, 1607402389504, 1490967855104, 4380756541440,
        14631844184064],
        "happiness_index": [6.94, 7.16, 6.66, 7.07, 6.38, 6.4, 7.23, 7.22, 5.87, 5.12]
    })

    # Instantiate a LLM
    from pandasai.llm.openai import OpenAI
    llm = OpenAI(api_token="YOUR_API_TOKEN")

    pandas_ai = PandasAI(llm)
    pandas_ai(df, prompt='Which are the 5 happiest countries?')

    ```
"""

import ast
import io
import logging
import re
import uuid
import time
import subprocess
import sys
from contextlib import redirect_stdout
from typing import List, Optional, Union, Dict, Type
import importlib.metadata

__version__ = importlib.metadata.version(__package__ or __name__)
import astor
import pandas as pd
from .constants import (
    WHITELISTED_BUILTINS,
    WHITELISTED_LIBRARIES,
)
from .exceptions import BadImportError, LLMNotFoundError
from .helpers._optional import import_dependency
from .helpers.anonymizer import anonymize_dataframe_head
from .helpers.cache import Cache
from .helpers.notebook import Notebook
from .helpers.save_chart import add_save_chart
from .helpers.shortcuts import Shortcuts
from .llm.base import LLM
from .llm.langchain import LangchainLLM
from .middlewares.base import Middleware
from .middlewares.charts import ChartsMiddleware
from .prompts.base import Prompt
from .prompts.correct_error_prompt import CorrectErrorPrompt
from .prompts.correct_multiples_prompt import CorrectMultipleDataframesErrorPrompt
from .prompts.generate_python_code import GeneratePythonCodePrompt
from .prompts.generate_response import GenerateResponsePrompt
from .prompts.multiple_dataframes import MultipleDataframesPrompt


def get_version():
    """
    Get the version from the package metadata
    """
    from importlib.metadata import version

    return version("pandasai")


__version__ = get_version()


class PandasAI(Shortcuts):
    """
    PandasAI is a wrapper around a LLM to make dataframes conversational.


    This is an entry point of `pandasai` object. This class consists of methods
    to interface the LLMs with Pandas     dataframes. A pandas dataframe metadata i.e.
    df.head() and prompt is passed on to chosen LLMs API end point to generate a Python
    code to answer the questions asked. The resultant python code is run on actual data
    and answer is converted into a conversational form.

    Note:
        Do not include the `self` parameter in the ``Args`` section.
    Args:
        _llm (obj): LLMs option to be used for API access
        _verbose (bool, optional): To show the intermediate outputs e.g. python code
        generated and execution step on the prompt. Default to False
        _is_conversational_answer (bool, optional): Whether to return answer in
        conversational form. Default to False
        _enforce_privacy (bool, optional): Do not display the data on prompt in case of
        Sensitive data. Default to False
        _max_retries (int, optional): max no. of tries to generate code on failure.
        Default to 3
        _in_notebook (bool, optional): Whether to run code in notebook. Default to False
        _original_instructions (dict, optional): The dict of instruction to run. Default
        to None
        _cache (Cache, optional): Cache object to store the results. Default to None
        _enable_cache (bool, optional): Whether to enable cache. Default to True
        _auto_copy_code (bool, optional): Whether to copy the code. Default to False
        _logger (logging.Logger, optional): Logger object to log the messages. Default
        to None
        _logs (List[dict], optional): List of logs to be stored. Default to []
        _prompt_id (str, optional): Unique ID to differentiate calls. Default to None
        _middlewares (List[Middleware], optional): List of middlewares to run. Default
        to [ChartsMiddleware()]
        _additional_dependencies (List[dict], optional): List of additional dependencies
        to be added. Default to []
        _custom_whitelisted_dependencies (List[str], optional): List of custom
        whitelisted dependencies. Default to []
        last_code_generated (str, optional): Pass last Code if generated. Default to
        None
        last_code_executed (str, optional): Pass the last execution / run. Default to
        None
        code_output (str, optional): The code output if any. Default to None
        last_error (str, optional): Error of running code last time. Default to None
        prompt_id (str, optional): Unique ID to differentiate calls. Default to None


    Returns (str): Response to a Question related to Data

    """

    _llm: LLM
    _verbose: bool = False
    _is_conversational_answer: bool = False
    _enforce_privacy: bool = False
    _max_retries: int = 3
    _in_notebook: bool = False
    _original_instructions: dict = {
        "question": None,
        "df_head": None,
        "num_rows": None,
        "num_columns": None,
    }
    _cache: Cache = None
    _enable_cache: bool = True
    _auto_copy_code: bool = False
    _prompt_id: Optional[str] = None
    _middlewares: List[Middleware] = [ChartsMiddleware()]
    _additional_dependencies: List[dict] = []
    _custom_whitelisted_dependencies: List[str] = []
    _start_time: float = 0
    _enable_logging: bool = True
    _logger: logging.Logger = None
    _logs: List[str] = []
    last_code_generated: Optional[str] = None
    last_code_executed: Optional[str] = None
    code_output: Optional[str] = None
    last_error: Optional[str] = None

    def __init__(
        self,
        llm=None,
        conversational=False,
        verbose=False,
        enforce_privacy=False,
        save_charts=False,
        save_charts_path=None,
        enable_cache=True,
        middlewares=None,
        custom_whitelisted_dependencies=None,
        enable_logging=True,
        auto_copy_code=False,
        non_default_prompts: Optional[Dict[str, Type[Prompt]]] = None,
    ):
        """

        __init__ method of the Class PandasAI

        Args:
            llm (object): LLMs option to be used for API access. Default is None
            conversational (bool): Whether to return answer in conversational form.
            Default to False
            verbose (bool): To show the intermediate outputs e.g. python code
            generated and execution step on the prompt.  Default to False
            enforce_privacy (bool): Execute the codes with Privacy Mode ON.
            Default to False
            save_charts (bool): Save the charts generated in the notebook.
            Default to False
            enable_cache (bool): Enable the cache to store the results.
            Default to True
            middlewares (list): List of middlewares to be used. Default to None
            custom_whitelisted_dependencies (list): List of custom dependencies to
            be used. Default to None
            enable_logging (bool): Enable the logging. Default to True
            non_default_prompts (dict): Mapping from keys to replacement prompt classes.
            Used to override specific types of prompts. Defaults to None.
        """

        # configure the logging
        # noinspection PyArgumentList
        # https://stackoverflow.com/questions/61226587/pycharm-does-not-recognize-logging-basicconfig-handlers-argument
        if enable_logging:
            handlers = [logging.FileHandler("pandasai.log")]
        else:
            handlers = []

        if verbose:
            handlers.append(logging.StreamHandler(sys.stdout))
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=handlers,
        )
        self._logger = logging.getLogger(__name__)

        if llm is None:
            raise LLMNotFoundError(
                "An LLM should be provided to instantiate a PandasAI instance"
            )
        self._load_llm(llm)
        self._is_conversational_answer = conversational
        self._verbose = verbose
        self._enforce_privacy = enforce_privacy
        self._save_charts = save_charts
        self._save_charts_path = save_charts_path
        self._process_id = str(uuid.uuid4())
        self._auto_copy_code = auto_copy_code
        self._logs = []

        self._non_default_prompts = (
            {} if non_default_prompts is None else non_default_prompts
        )

        self.notebook = Notebook()
        self._in_notebook = self.notebook.in_notebook()

        self._enable_cache = enable_cache
        if self._enable_cache:
            self._cache = Cache()

        if middlewares is not None:
            self.add_middlewares(*middlewares)

        if custom_whitelisted_dependencies is not None:
            self._custom_whitelisted_dependencies = custom_whitelisted_dependencies

    def _load_llm(self, llm):
        """
        Check if it is a PandasAI LLM or a Langchain LLM.
        If it is a Langchain LLM, wrap it in a PandasAI LLM.

        Args:
            llm (object): LLMs option to be used for API access

        Raises:
            BadImportError: If the LLM is a Langchain LLM but the langchain package
            is not installed
        """

        try:
            llm.is_pandasai_llm()
        except AttributeError:
            llm = LangchainLLM(llm)

        self._llm = llm

    def conversational_answer(self, question: str, answer: str) -> str:
        """
        Returns the answer in conversational form about the resultant data.

        Args:
            question (str): A question in Conversational form
            answer (str): A summary / resultant Data

        Returns (str): Response

        """

        if self._enforce_privacy:
            # we don't want to send potentially sensitive data to the LLM server
            # if the user has set enforce_privacy to True
            return answer

        generate_response_instruction = self._non_default_prompts.get(
            "generate_response", GenerateResponsePrompt
        )(question=question, answer=answer)
        return self._llm.call(generate_response_instruction, "")

    def run(
        self,
        data_frame: Union[pd.DataFrame, List[pd.DataFrame]],
        prompt: str,
        is_conversational_answer: bool = None,
        show_code: bool = False,
        anonymize_df: bool = True,
        use_error_correction_framework: bool = True,
    ) -> Union[str, pd.DataFrame]:
        """
        Run the PandasAI to make Dataframes Conversational.

        Args:
            data_frame (Union[pd.DataFrame, List[pd.DataFrame]]): A pandas Dataframe
            prompt (str): A prompt to query about the Dataframe
            is_conversational_answer (bool): Whether to return answer in conversational
            form. Default to False
            show_code (bool): To show the intermediate python code generated on the
            prompt. Default to False
            anonymize_df (bool): Running the code with Sensitive Data. Default to True
            use_error_correction_framework (bool): Turn on Error Correction mechanism.
            Default to True

        Returns (str): Answer to the Input Questions about the DataFrame

        """

        self._start_time = time.time()

        self.log(f"Question: {prompt}")
        self.log(f"Running PandasAI with {self._llm.type} LLM...")

        self._prompt_id = str(uuid.uuid4())
        self.log(f"Prompt ID: {self._prompt_id}")

        try:
            if self._enable_cache and self._cache and self._cache.get(prompt):
                self.log("Using cached response")
                code = self._cache.get(prompt)
            else:
                rows_to_display = 0 if self._enforce_privacy else 5

                multiple: bool = isinstance(data_frame, list)

                if multiple:
                    heads = [
                        anonymize_dataframe_head(dataframe)
                        if anonymize_df
                        else dataframe.head(rows_to_display)
                        for dataframe in data_frame
                    ]

                    multiple_dataframes_instruction = self._non_default_prompts.get(
                        "multiple_dataframes", MultipleDataframesPrompt
                    )
                    code = self._llm.generate_code(
                        multiple_dataframes_instruction(dataframes=heads),
                        prompt,
                    )

                    self._original_instructions = {
                        "question": prompt,
                        "df_head": heads,
                    }

                else:
                    df_head = data_frame.head(rows_to_display)
                    if anonymize_df:
                        df_head = anonymize_dataframe_head(df_head)
                    df_head = df_head.to_csv(index=False)

                    generate_code_instruction = self._non_default_prompts.get(
                        "generate_python_code", GeneratePythonCodePrompt
                    )(
                        prompt=prompt,
                        df_head=df_head,
                        num_rows=data_frame.shape[0],
                        num_columns=data_frame.shape[1],
                    )
                    code = self._llm.generate_code(
                        generate_code_instruction,
                        prompt,
                    )

                    self._original_instructions = {
                        "question": prompt,
                        "df_head": df_head,
                        "num_rows": data_frame.shape[0],
                        "num_columns": data_frame.shape[1],
                    }

                self.last_code_generated = code
                self.log(
                    f"""
                        Code generated:
                        ```
                        {code}
                        ```
                    """
                )

                if self._enable_cache and self._cache:
                    self._cache.set(prompt, code)

            if show_code and self._in_notebook:
                self.notebook.create_new_cell(code)

            for middleware in self._middlewares:
                code = middleware(code)

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
                answer = self.conversational_answer(prompt, answer)
                self.log(f"Conversational answer: {answer}")

            self.log(f"Executed in: {time.time() - self._start_time}s")

            return answer
        except Exception as exception:
            self.last_error = str(exception)
            print(exception)
            return (
                "Unfortunately, I was not able to answer your question, "
                "because of the following error:\n"
                f"\n{exception}\n"
            )

    def add_middlewares(self, *middlewares: List[Middleware]):
        """
        Add middlewares to PandasAI instance.

        Args:
            *middlewares: A list of middlewares

        """
        self._middlewares.extend(middlewares)

    def clear_cache(self):
        """
        Clears the cache of the PandasAI instance.
        """
        if self._cache:
            self._cache.clear()

    def __call__(
        self,
        data_frame: Union[pd.DataFrame, List[pd.DataFrame]],
        prompt: str,
        is_conversational_answer: bool = None,
        show_code: bool = False,
        anonymize_df: bool = True,
        use_error_correction_framework: bool = True,
    ) -> Union[str, pd.DataFrame]:
        """
        __call__ method of PandasAI class. It calls the `run` method.

        Args:
            data_frame:
            prompt:
            is_conversational_answer:
            show_code:
            anonymize_df:
            use_error_correction_framework:

        Returns (str): Answer to the Input Questions about the DataFrame.

        """

        return self.run(
            data_frame,
            prompt,
            is_conversational_answer,
            show_code,
            anonymize_df,
            use_error_correction_framework,
        )

    def _check_imports(self, node: Union[ast.Import, ast.ImportFrom]):
        """
        Add whitelisted imports to _additional_dependencies.

        Args:
            node (object): ast.Import or ast.ImportFrom

        Raises:
            BadImportError: If the import is not whitelisted

        """
        if isinstance(node, ast.Import):
            module = node.names[0].name
        else:
            module = node.module

        library = module.split(".")[0]

        if library == "pandas":
            return

        if library in WHITELISTED_LIBRARIES + self._custom_whitelisted_dependencies:
            for alias in node.names:
                self._additional_dependencies.append(
                    {
                        "module": module,
                        "name": alias.name,
                        "alias": alias.asname or alias.name,
                    }
                )
            return

        if library not in WHITELISTED_BUILTINS:
            raise BadImportError(library)

    def _is_df_overwrite(self, node: ast.stmt) -> bool:
        """
        Remove df declarations from the code to prevent malicious code execution.

        Args:
            node (object): ast.stmt

        Returns (bool):

        """

        return (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and re.match(r"df\d{0,2}$", node.targets[0].id)
        )

    def _clean_code(self, code: str) -> str:
        """
        A method to clean the code to prevent malicious code execution

        Args:
            code(str): A python code

        Returns (str): Returns a Clean Code String

        """

        tree = ast.parse(code)

        new_body = []

        # clear recent optional dependencies
        self._additional_dependencies = []

        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                self._check_imports(node)
                continue
            if self._is_df_overwrite(node):
                continue
            new_body.append(node)

        new_tree = ast.Module(body=new_body)
        return astor.to_source(new_tree, pretty_source=lambda x: "".join(x)).strip()

    def _get_environment(self) -> dict:
        """
        Returns the environment for the code to be executed.

        Returns (dict): A dictionary of environment variables
        """

        return {
            "pd": pd,
            **{
                lib["alias"]: getattr(import_dependency(lib["module"]), lib["name"])
                if hasattr(import_dependency(lib["module"]), lib["name"])
                else import_dependency(lib["module"])
                for lib in self._additional_dependencies
            },
            "__builtins__": {
                **{builtin: __builtins__[builtin] for builtin in WHITELISTED_BUILTINS},
            },
        }

    def _retry_run_code(self, code: str, e: Exception, multiple: bool = False):
        """
        A method to retry the code execution with error correction framework.

        Args:
            code (str): A python code
            e (Exception): An exception
            multiple (bool): A boolean to indicate if the code is for multiple
            dataframes

        Returns (str): A python code
        """

        if multiple:
            error_correcting_instruction = self._non_default_prompts.get(
                "correct_multiple_dataframes_error",
                CorrectMultipleDataframesErrorPrompt,
            )(
                code=code,
                error_returned=e,
                question=self._original_instructions["question"],
                df_head=self._original_instructions["df_head"],
            )

        else:
            error_correcting_instruction = self._non_default_prompts.get(
                "correct_error", CorrectErrorPrompt
            )(
                code=code,
                error_returned=e,
                question=self._original_instructions["question"],
                df_head=self._original_instructions["df_head"],
                num_rows=self._original_instructions["num_rows"],
                num_columns=self._original_instructions["num_columns"],
            )

        return self._llm.generate_code(error_correcting_instruction, "")

    def run_code(
        self,
        code: str,
        data_frame: pd.DataFrame,
        use_error_correction_framework: bool = True,
    ) -> str:
        """
        A method to execute the python code generated by LLMs to answer the question
        about the input dataframe. Run the code in the current context and return the
        result.

        Args:
            code (str): A python code to execute
            data_frame (pd.DataFrame): A full Pandas DataFrame
            use_error_correction_framework (bool): Turn on Error Correction mechanism.
            Default to True

        Returns (str): String representation of the result of the code execution.

        """

        multiple: bool = isinstance(data_frame, list)

        # Add save chart code
        if self._save_charts:
            code = add_save_chart(
                code, self._prompt_id, self._save_charts_path, not self._verbose
            )

        # Get the code to run removing unsafe imports and df overwrites
        code_to_run = self._clean_code(code)

        # Copy the code to clipboard
        if self._auto_copy_code:
            self.copy_to_clipboard(code_to_run)

        self.last_code_executed = code_to_run
        self.log(
            f"""
Code running:
```
{code_to_run}
```"""
        )

        environment: dict = self._get_environment()

        if multiple:
            environment.update(
                {f"df{i}": dataframe for i, dataframe in enumerate(data_frame, start=1)}
            )
        else:
            environment["df"] = data_frame

        # Redirect standard output to a StringIO buffer
        with redirect_stdout(io.StringIO()) as output:
            count = 0
            while count < self._max_retries:
                try:
                    # Execute the code
                    exec(code_to_run, environment)
                    code = code_to_run
                    break
                except Exception as e:
                    if not use_error_correction_framework:
                        raise e

                    count += 1

                    code_to_run = self._retry_run_code(code, e, multiple)

        captured_output = output.getvalue().strip()
        if code.count("print(") > 1:
            return captured_output

        # Evaluate the last line and return its value or the captured output
        # We do this because we want to return the right value and the right
        # type of the value. For example, if the last line is `df.head()`, we
        # want to return the head of the dataframe, not the captured output.
        lines = code.strip().split("\n")
        last_line = lines[-1].strip()

        match = re.match(r"^print\((.*)\)$", last_line)
        if match:
            last_line = match.group(1)

        try:
            result = eval(last_line, environment)

            # In some cases, the result is a tuple of values. For example, when
            # the last line is `print("Hello", "World")`, the result is a tuple
            # of two strings. In this case, we want to return a string
            if isinstance(result, tuple):
                result = " ".join([str(element) for element in result])

            return result
        except Exception:
            return captured_output

    def log(self, message: str):
        """Log a message"""
        self._logger.info(message)
        self._logs.append(message)

    def copy_to_clipboard(self, text: str):
        """
        Copy text to clipboard

        Args:
            text (str): Text to copy to clipboard
        """
        if sys.platform == "win32":
            command = "echo " + text.strip() + "| clip"
            subprocess.run(command, shell=True)
        elif sys.platform == "darwin":
            command = "echo " + text.strip() + "| pbcopy"
            subprocess.run(command, shell=True)
        elif sys.platform.startswith("linux"):
            command = "echo " + text.strip() + "| xclip -selection clipboard"
            subprocess.run(command, shell=True)
        else:
            raise Exception("Can't copy to clipboard, unsupported OS.")

    def paste_from_clipboard(self) -> str:
        """
        Paste text from clipboard

        Returns (str): Text from clipboard
        """
        if sys.platform == "win32":
            command = 'powershell.exe -command "Get-Clipboard"'
            result = subprocess.run(command, capture_output=True, text=True, shell=True)
            return result.stdout.strip()
        elif sys.platform == "darwin":
            command = "pbpaste"
            result = subprocess.run(command, capture_output=True, text=True, shell=True)
            return result.stdout.strip()
        elif sys.platform.startswith("linux"):
            command = "xclip -selection clipboard -o"
            result = subprocess.run(command, capture_output=True, text=True, shell=True)
            return result.stdout.strip()
        else:
            raise Exception("Can't paste from clipboard, unsupported OS.")

    @property
    def logs(self) -> List[str]:
        """Return the logs"""
        return self._logs

    def process_id(self) -> str:
        """Return the id of this PandasAI object."""
        return self._process_id

    @property
    def last_prompt_id(self) -> str:
        """Return the id of the last prompt that was run."""
        if self._prompt_id is None:
            raise ValueError("Pandas AI has not been run yet.")
        return self._prompt_id

    @property
    def last_prompt(self) -> str:
        """Return the last prompt that was executed."""
        if self._llm:
            return self._llm.last_prompt
