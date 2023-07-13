"""Unit tests for the PandasAI class"""

import sys
from datetime import date
from typing import Optional
from unittest.mock import Mock, patch
from uuid import UUID

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pytest

from pandasai import PandasAI, Prompt
from pandasai.exceptions import BadImportError, LLMNotFoundError, NoCodeFoundError
from pandasai.llm.fake import FakeLLM
from pandasai.middlewares.base import Middleware
from langchain.llms import OpenAI


class TestPandasAI:
    """Unit tests for the PandasAI class"""

    @pytest.fixture
    def llm(self, output: Optional[str] = None):
        return FakeLLM(output=output)

    @pytest.fixture
    def pandasai(self, llm):
        return PandasAI(llm, enable_cache=False)

    @pytest.fixture
    def sample_df(self, llm):
        return pd.DataFrame(
            {
                "country": [
                    "United States",
                    "United Kingdom",
                    "France",
                    "Germany",
                    "Italy",
                    "Spain",
                    "Canada",
                    "Australia",
                    "Japan",
                    "China",
                ],
                "gdp": [
                    19294482071552,
                    2891615567872,
                    2411255037952,
                    3435817336832,
                    1745433788416,
                    1181205135360,
                    1607402389504,
                    1490967855104,
                    4380756541440,
                    14631844184064,
                ],
                "happiness_index": [
                    6.94,
                    7.16,
                    6.66,
                    7.07,
                    6.38,
                    6.4,
                    7.23,
                    7.22,
                    5.87,
                    5.12,
                ],
            }
        )

    @pytest.fixture
    def test_middleware(self):
        class TestMiddleware(Middleware):
            def run(self, code: str) -> str:
                return "print('Overwritten by middleware')"

        return TestMiddleware

    def test_init(self, pandasai):
        assert pandasai._llm is not None
        assert pandasai._is_conversational_answer is False
        assert pandasai._verbose is False

    def test_init_without_llm(self):
        with pytest.raises(LLMNotFoundError):
            PandasAI()

    def test_conversational_answer(self, pandasai, llm):
        result = "2"
        llm._output = result
        assert pandasai.conversational_answer("What is the sum of 1 + 1?", 2) == result

    def test_run(self, pandasai, llm):
        df = pd.DataFrame()
        llm._output = "1"
        assert pandasai.run(df, "What number comes before 2?") == 1

    def test_run_with_conversational_answer(self, pandasai, llm):
        df = pd.DataFrame()
        llm._output = "1 + 1"
        assert (
            pandasai.run(df, "What is the sum of 1 + 1?", is_conversational_answer=True)
            == "1 + 1"
        )

    def test_run_with_non_conversational_answer(self, pandasai, llm):
        df = pd.DataFrame()
        llm._output = "1 + 1"
        assert (
            pandasai.run(
                df, "What is the sum of 1 + 1?", is_conversational_answer=False
            )
            == 2
        )

    def test_run_with_verbose(self, pandasai):
        df = pd.DataFrame()
        pandasai._verbose = True

        # mock print function
        with patch("builtins.print") as mock_print:
            pandasai.run(df, "What number comes before 2?")
            mock_print.assert_called()

    def test_run_without_verbose(self, pandasai, llm):
        df = pd.DataFrame()
        pandasai._verbose = False
        llm._output = "1"

        # mock print function
        with patch("builtins.print") as mock_print:
            pandasai.run(df, "What number comes before 2?")
            mock_print.assert_not_called()

    def test_run_code(self, pandasai):
        df = pd.DataFrame({"a": [1, 2, 3]})
        code = """
df["b"] = df["a"] + 1
df
"""
        pandasai._llm._output = code
        assert pandasai.run_code(code, df).equals(
            pd.DataFrame({"a": [1, 2, 3], "b": [2, 3, 4]})
        )

    def test_run_code_for_calculations(self, pandasai):
        df = pd.DataFrame()
        assert pandasai.run_code("1 + 1", df) == 2
        assert pandasai.last_code_executed == "1 + 1"

    def test_run_code_invalid_code(self):
        df = pd.DataFrame()
        with pytest.raises(Exception):
            PandasAI().run_code("1 +", df, use_error_correction_framework=False)

    def test_run_code_with_print(self, pandasai):
        df = pd.DataFrame()
        assert pandasai.run_code("print(1 + 1)", df) == 2

    def test_conversational_answer_with_privacy_enforcement(self, pandasai, llm):
        pandasai._enforce_privacy = True
        llm.call = Mock(return_value="The answer is 2")
        assert pandasai.conversational_answer("How much does 1 + 1 do?", 2) == 2
        llm.call.assert_not_called()

    def test_conversational_answer_without_privacy_enforcement(self, pandasai, llm):
        pandasai._enforce_privacy = False
        llm.call = Mock(return_value="The answer is 2")
        assert (
            pandasai.conversational_answer("How much does 1 + 1 do?", 2)
            == "The answer is 2"
        )
        llm.call.assert_called()

    def test_run_with_privacy_enforcement(self, pandasai):
        df = pd.DataFrame({"country": ["United States", "United Kingdom", "France"]})
        pandasai._enforce_privacy = True
        pandasai._is_conversational_answer = True

        expected_prompt = f"""
Today is {date.today()}.
You are provided with a pandas dataframe (df) with 3 rows and 1 columns.
This is the metadata of the dataframe:
country
.

When asked about the data, your response should include a python code that describes the dataframe `df`.
Using the provided dataframe, df, return the python code to get the answer to the following question:
How many countries are in the dataframe?

Code:
"""  # noqa: E501
        pandasai.run(df, "How many countries are in the dataframe?")
        last_prompt = pandasai._llm.last_prompt
        if sys.platform.startswith('win'):
            last_prompt = last_prompt.replace('\r\n', '\n')
        assert last_prompt == expected_prompt

    def test_run_with_anonymized_df(self, pandasai):
        df = pd.DataFrame(
            {
                "Phone Number": ["(743) 226-2382", "+1 123456789", "0002223334"],
                "Email": [
                    "linda55@nguyen-williams.info",
                    "oliverdouglas@lee-harris.biz",
                    "sara41@mitchell-rodriguez.com",
                ],
                "Name": [
                    "Rachel Davis",
                    "Nathan Richards",
                    "Monica Scott",
                ],
                "Age": [
                    54,
                    21,
                    27,
                ],
                "Credit Card Number": [
                    "5416931670890256",
                    "3109-2849-2297-7926",
                    "4795 0612 5882 4558",
                ],
            }
        )
        pandasai._is_conversational_answer = True
        expected_non_anonymized_data_frame_substring = """
This is the result of `print(df.head(5))`:
     Phone Number                          Email             Name  Age   Credit Card Number
0  (743) 226-2382   linda55@nguyen-williams.info     Rachel Davis   54     5416931670890256
1    +1 123456789   oliverdouglas@lee-harris.biz  Nathan Richards   21  3109-2849-2297-7926
2      0002223334  sara41@mitchell-rodriguez.com     Monica Scott   27  4795 0612 5882 4558.
"""  # noqa: E501
        pandasai.run(df, "How many people are in the dataframe?", anonymize_df=True)
        assert (
            expected_non_anonymized_data_frame_substring
            not in pandasai._llm.last_prompt
        )

    def test_run_without_privacy_enforcement(self, pandasai):
        df = pd.DataFrame({"country": ["United States", "United Kingdom", "France"]})
        pandasai._enforce_privacy = False
        pandasai._is_conversational_answer = False

        expected_prompt = f"""
Today is {date.today()}.
You are provided with a pandas dataframe (df) with 3 rows and 1 columns.
This is the metadata of the dataframe:
country
United States
United Kingdom
France
.

When asked about the data, your response should include a python code that describes the dataframe `df`.
Using the provided dataframe, df, return the python code to get the answer to the following question:
How many countries are in the dataframe?

Code:
"""  # noqa: E501
        pandasai.run(df, "How many countries are in the dataframe?", anonymize_df=False)
        last_prompt = pandasai._llm.last_prompt
        if sys.platform.startswith('win'):
            last_prompt = last_prompt.replace('\r\n', '\n')
        assert last_prompt == expected_prompt

    def test_run_with_print_at_the_end(self, pandasai, llm):
        code = """
result = {'happiness': 0.5, 'gdp': 0.8}
print(result)"""
        llm._output = code
        pandasai.run_code(code, pd.DataFrame())

    def test_extract_code(self, pandasai):
        code = """```python
result = {'happiness': 0.5, 'gdp': 0.8}
print(result)```"""
        assert (
            pandasai._llm._extract_code(code)
            == "result = {'happiness': 0.5, 'gdp': 0.8}\nprint(result)"
        )

        code = """```
result = {'happiness': 1, 'gdp': 0.43}```"""
        assert (
            pandasai._llm._extract_code(code)
            == "result = {'happiness': 1, 'gdp': 0.43}"
        )

    def test_clean_code_remove_builtins(self, pandasai):
        builtins_code = """
import set
print(set([1, 2, 3]))
"""
        pandasai._llm._output = builtins_code
        assert pandasai.run_code(builtins_code, pd.DataFrame()) == {1, 2, 3}
        assert pandasai.last_code_executed == "print(set([1, 2, 3]))"

    def test_clean_code_remove_environment_defaults(self, pandasai):
        pandas_code = """
import pandas as pd
print(df.size)
"""
        pandasai._llm._output = pandas_code
        pandasai.run_code(pandas_code, pd.DataFrame())
        assert pandasai.last_code_executed == "print(df.size)"

    def test_clean_code_whitelist_import(self, pandasai):
        """Test that an installed whitelisted library is added to the environment."""
        safe_code = """
import numpy as np
np.array()
"""
        pandasai._llm._output = safe_code
        assert pandasai.run_code(safe_code, pd.DataFrame()) == ""
        assert pandasai.last_code_executed == "np.array()"

    def test_clean_code_whitelist_import_from(self, pandasai):
        """Test that an import from statement is added to the environment."""
        optional_code = """
from numpy import array
array()
"""
        pandasai._llm._output = optional_code
        assert pandasai.run_code(optional_code, pd.DataFrame()) == ""

    def test_clean_code_whitelist_import_from_multiple(self, pandasai):
        """Test that multiple imports from a library are added to the environment."""
        optional_code = """
from numpy import array, zeros
array()
"""
        pandasai._llm._output = optional_code
        assert pandasai.run_code(optional_code, pd.DataFrame()) == ""

    def test_clean_code_raise_bad_import_error(self, pandasai):
        malicious_code = """
import os
print(os.listdir())
"""
        pandasai._llm._output = malicious_code
        with pytest.raises(BadImportError):
            pandasai.run_code(malicious_code, pd.DataFrame())

    def test_clean_code_raise_import_error(self, pandasai):
        """Test that an ImportError is raised when
        the code contains an import statement for an optional library."""
        optional_code = """
import seaborn as sns
print(df)
"""
        pandasai._llm._output = optional_code

        # patch the import of seaborn to raise an ImportError
        with pytest.raises(ImportError):
            with patch.dict("sys.modules", {"seaborn": None}):
                pandasai.run_code(optional_code, pd.DataFrame())

    def test_remove_df_overwrites(self, pandasai):
        malicious_code = """
df = pd.DataFrame([1,2,3])
print(df)
"""
        pandasai._llm._output = malicious_code
        pandasai.run_code(malicious_code, pd.DataFrame())
        assert pandasai.last_code_executed == "print(df)"

    def test_exception_handling(self, pandasai):
        pandasai.run_code = Mock(
            side_effect=NoCodeFoundError("No code found in the answer.")
        )

        result = pandasai(pd.DataFrame(), "How many countries are in the dataframe?")
        assert result == (
            "Unfortunately, I was not able to answer your question, "
            "because of the following error:\n"
            "\nNo code found in the answer.\n"
        )
        assert pandasai.last_error == "No code found in the answer."

    def test_cache(self, llm):
        pandasai = PandasAI(llm=llm)
        pandasai._llm.call = Mock(return_value='print("Hello world")')
        assert pandasai._cache.get("How many countries are in the dataframe?") is None
        pandasai(
            pd.DataFrame(),
            "How many countries are in the dataframe?",
        )
        assert (
            pandasai._cache.get("How many countries are in the dataframe?")
            == 'print("Hello world")'
        )
        pandasai(
            pd.DataFrame(),
            "How many countries are in the dataframe?",
        )
        assert pandasai._llm.call.call_count == 1
        pandasai._cache.delete("How many countries are in the dataframe?")

    def test_process_id(self, pandasai):
        process_id = pandasai.process_id()
        assert isinstance(UUID(process_id, version=4), UUID)

    def test_last_prompt_id(self, pandasai):
        pandasai(pd.DataFrame(), "How many countries are in the dataframe?")
        prompt_id = pandasai.last_prompt_id
        assert isinstance(UUID(prompt_id, version=4), UUID)

    def test_last_prompt_id_no_prompt(self, pandasai):
        with pytest.raises(ValueError):
            pandasai.last_prompt_id

    def test_add_middlewares(self, pandasai, test_middleware):
        middleware = test_middleware()
        pandasai.add_middlewares(middleware)
        assert pandasai._middlewares[len(pandasai._middlewares) - 1] == middleware

    def test_middlewares(self, pandasai, test_middleware):
        middleware = test_middleware()
        pandasai._middlewares = [middleware]
        assert pandasai._middlewares == [middleware]
        assert (
            pandasai(pd.DataFrame(), "How many countries are in the dataframe?")
            == "Overwritten by middleware"
        )
        assert middleware.has_run

    def test_custom_whitelisted_dependencies(self, pandasai):
        code = """
import my_custom_library
my_custom_library.do_something()
"""
        pandasai._llm._output = code

        with pytest.raises(BadImportError):
            pandasai._clean_code(code)

        pandasai._custom_whitelisted_dependencies = ["my_custom_library"]
        assert pandasai._clean_code(code) == "my_custom_library.do_something()"

    def test_load_llm_with_pandasai_llm(self, pandasai, llm):
        pandasai._load_llm(llm)
        assert pandasai._llm == llm

    def test_load_llm_with_langchain_llm(self, pandasai):
        langchain_llm = OpenAI(openai_api_key="fake_key")

        pandasai._load_llm(langchain_llm)
        assert pandasai._llm._langchain_llm == langchain_llm

    def test_get_environment(self, pandasai):
        pandasai._additional_dependencies = [
            {"name": "pyplot", "alias": "plt", "module": "matplotlib"},
            {"name": "numpy", "alias": "np", "module": "numpy"},
        ]
        assert pandasai._get_environment() == {
            "pd": pd,
            "plt": plt,
            "np": np,
            "__builtins__": {
                "abs": abs,
                "all": all,
                "any": any,
                "ascii": ascii,
                "bin": bin,
                "bool": bool,
                "bytearray": bytearray,
                "bytes": bytes,
                "callable": callable,
                "chr": chr,
                "classmethod": classmethod,
                "complex": complex,
                "delattr": delattr,
                "dict": dict,
                "dir": dir,
                "divmod": divmod,
                "enumerate": enumerate,
                "filter": filter,
                "float": float,
                "format": format,
                "frozenset": frozenset,
                "getattr": getattr,
                "hasattr": hasattr,
                "hash": hash,
                "help": help,
                "hex": hex,
                "id": id,
                "input": input,
                "int": int,
                "isinstance": isinstance,
                "issubclass": issubclass,
                "iter": iter,
                "len": len,
                "list": list,
                "locals": locals,
                "map": map,
                "max": max,
                "memoryview": memoryview,
                "min": min,
                "next": next,
                "object": object,
                "oct": oct,
                "open": open,
                "ord": ord,
                "pow": pow,
                "print": print,
                "property": property,
                "range": range,
                "repr": repr,
                "reversed": reversed,
                "round": round,
                "set": set,
                "setattr": setattr,
                "slice": slice,
                "sorted": sorted,
                "staticmethod": staticmethod,
                "str": str,
                "sum": sum,
                "super": super,
                "tuple": tuple,
                "type": type,
                "vars": vars,
                "zip": zip,
            },
        }

    def test_retry_on_error_with_single_df(self, pandasai, sample_df):
        code = 'print("Hello world")'

        pandasai._original_instructions = {
            "question": "Print hello world",
            "df_head": sample_df.head(),
            "num_rows": 10,
            "num_columns": 3,
        }
        pandasai._retry_run_code(code, e=Exception("Test error"), multiple=False)
        assert (
            pandasai.last_prompt
            == f"""
Today is {date.today()}.
You are provided with a pandas dataframe (df) with 10 rows and 3 columns.
This is the metadata of the dataframe:
          country             gdp  happiness_index
0   United States  19294482071552             6.94
1  United Kingdom   2891615567872             7.16
2          France   2411255037952             6.66
3         Germany   3435817336832             7.07
4           Italy   1745433788416             6.38.

The user asked the following question:
Print hello world

You generated this python code:
print("Hello world")

It fails with the following error:
Test error

Correct the python code and return a new python code (do not import anything) that fixes the above mentioned error. Do not generate the same code again.


Code:
"""  # noqa: E501
        )

    def test_retry_on_error_with_multiple_df(self, pandasai, sample_df):
        code = 'print("Hello world")'

        pandasai._original_instructions = {
            "question": "Print hello world",
            "df_head": [sample_df.head()],
            "num_rows": 10,
            "num_columns": 3,
        }
        pandasai._retry_run_code(code, e=Exception("Test error"), multiple=True)
        assert (
            pandasai.last_prompt
            == """
You are provided with the following pandas dataframes:
Dataframe df1, with 5 rows and 3 columns.
This is the metadata of the dataframe df1:
          country             gdp  happiness_index
0   United States  19294482071552             6.94
1  United Kingdom   2891615567872             7.16
2          France   2411255037952             6.66
3         Germany   3435817336832             7.07
4           Italy   1745433788416             6.38
The user asked the following question:
Print hello world

You generated this python code:
print("Hello world")

It fails with the following error:
Test error

Correct the python code and return a new python code (do not import anything) that fixes the above mentioned error. Do not generate the same code again.


Code:
"""  # noqa: E501
        )

    def test_catches_multiple_prints(self, pandasai):
        code = """
print("Hello world")
print("Hello world again")
"""

        response = pandasai.run_code(code, pd.DataFrame())
        assert (
            response
            == """Hello world
Hello world again"""
        )

    def test_catches_print_with_multiple_args(self, pandasai):
        code = """name = "John"
print('Hello', name)"""

        response = pandasai.run_code(code, pd.DataFrame())
        assert response == "Hello John"

    def test_shortcut(self, pandasai):
        pandasai.run = Mock(return_value="Hello world")
        pandasai.clean_data(pd.DataFrame())
        pandasai.run.assert_called_once()

    def test_shortcut_with_multiple_df(self, pandasai):
        pandasai.run = Mock(return_value="Hello world")
        pandasai.clean_data([pd.DataFrame(), pd.DataFrame()])
        pandasai.run.assert_called_once()

    def test_replace_generate_code_prompt(self, llm):
        replacement_prompt = "{num_rows} | {num_columns} | {df_head} | ".format

        pai = PandasAI(
            llm,
            non_default_prompts={"generate_python_code": replacement_prompt},
            enable_cache=False,
        )
        question = "Will this work?"
        df = pd.DataFrame()
        pai(df, question, use_error_correction_framework=False)
        expected_last_prompt = (
            str(
                replacement_prompt(
                    num_rows=df.shape[0],
                    num_columns=df.shape[1],
                    df_head=df.head().to_csv(index=False),
                )
            )
            + question
            + "\n\nCode:\n"
        )
        assert llm.last_prompt == expected_last_prompt

    def test_replace_correct_error_prompt(self, llm):
        replacement_prompt = (
            "{num_rows} | {num_columns} | {df_head} | "
            "{question} | {code} | {error_returned} |".format
        )

        pai = PandasAI(
            llm,
            non_default_prompts={"correct_error": replacement_prompt},
            enable_cache=False,
        )

        df = pd.DataFrame()

        erroneous_code = "a"
        question = "Will this work?"
        num_rows = df.shape[0]
        num_columns = df.shape[1]
        df_head = df.head()

        pai._original_instructions["question"] = question
        pai._original_instructions["df_head"] = df_head
        pai._original_instructions["num_rows"] = num_rows
        pai._original_instructions["num_columns"] = num_columns
        pai.run_code(erroneous_code, df, use_error_correction_framework=True)

        expected_last_prompt = (
            str(
                replacement_prompt(
                    code=erroneous_code,
                    error_returned="name 'a' is not defined",
                    question=question,
                    df_head=df_head,
                    num_rows=num_rows,
                    num_columns=num_columns,
                )
            )
            + ""  # "prompt" parameter passed as empty string
            + "\n\nCode:\n"
        )
        assert llm.last_prompt == expected_last_prompt

    def test_replace_multiple_dataframes_prompt(self, llm):
        class ReplacementPrompt(Prompt):
            text = ""

            def __init__(self, dataframes, **kwargs):
                super().__init__(
                    **kwargs,
                )
                for df in dataframes:
                    self.text += f"\n{df}\n"

        pai = PandasAI(
            llm,
            non_default_prompts={"multiple_dataframes": ReplacementPrompt},
            enable_cache=False,
        )
        question = "Will this work?"
        dataframes = [pd.DataFrame(), pd.DataFrame()]

        pai(
            dataframes,
            question,
            anonymize_df=False,
            use_error_correction_framework=False,
        )

        heads = [dataframe.head(5) for dataframe in dataframes]

        expected_last_prompt = (
            str(ReplacementPrompt(dataframes=heads)) + question + "\n\nCode:\n"
        )
        assert llm.last_prompt == expected_last_prompt

    def test_replace_generate_response_prompt(self, llm):
        replacement_prompt = "{question} | {answer} | ".format

        pai = PandasAI(
            llm,
            non_default_prompts={"generate_response": replacement_prompt},
            enable_cache=False,
        )
        question = "Will this work?"
        answer = "No it won't"
        pai.conversational_answer(question, answer)
        expected_last_prompt = (
            str(replacement_prompt(question=question, answer=answer))
            + ""  # "value" parameter passed as empty string
            + ""  # "suffix" parameter defaults to empty string
        )
        assert llm.last_prompt == expected_last_prompt

    def test_replace_correct_multiple_dataframes_error_prompt(self, llm):
        replacement_prompt = (
            "{df_head} | " "{question} | {code} | {error_returned} |".format
        )

        pai = PandasAI(
            llm,
            non_default_prompts={
                "correct_multiple_dataframes_error": replacement_prompt
            },
            enable_cache=False,
        )

        dataframes = [pd.DataFrame(), pd.DataFrame()]

        erroneous_code = "a"
        question = "Will this work?"
        heads = [dataframe.head(5) for dataframe in dataframes]

        pai._original_instructions["question"] = question
        pai._original_instructions["df_head"] = heads
        pai.run_code(erroneous_code, dataframes, use_error_correction_framework=True)

        expected_last_prompt = (
            str(
                replacement_prompt(
                    code=erroneous_code,
                    error_returned="name 'a' is not defined",
                    question=question,
                    df_head=heads,
                )
            )
            + ""  # "prompt" parameter passed as empty string
            + "\n\nCode:\n"
        )
        assert llm.last_prompt == expected_last_prompt

    def test_saves_logs(self, llm):
        pandas_ai = PandasAI(llm)
        assert pandas_ai.logs == []
        pandas_ai.log("a")
        pandas_ai.log("b")
        assert pandas_ai.logs == [
            "a",
            "b",
        ]

    def test_copy_to_clipboard(self, llm):
        pandas_ai = PandasAI(llm)
        pandas_ai.copy_to_clipboard("a")
        assert pandas_ai.paste_from_clipboard() == "a"
