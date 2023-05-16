"""Unit tests for the PandasAI class"""

from datetime import date
from typing import Optional
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from pandasai import PandasAI
from pandasai.exceptions import LLMNotFoundError
from pandasai.llm.fake import FakeLLM


class TestPandasAI:
    """Unit tests for the PandasAI class"""

    # pylint: disable=missing-function-docstring protected-access invalid-name

    @pytest.fixture
    def llm(self, output: Optional[str] = None):
        return FakeLLM(output=output)

    @pytest.fixture
    def pandasai(self, llm):
        return PandasAI(llm)

    def test_init(self, pandasai):
        assert pandasai._llm is not None
        assert pandasai._is_conversational_answer is True
        assert pandasai._verbose is False

    def test_init_without_llm(self):
        with pytest.raises(LLMNotFoundError):
            PandasAI()

    def test_conversational_answer(self, pandasai, llm):
        result = "2"
        llm._output = result
        assert (
            pandasai.conversational_answer("What is the sum of 1 + 1?", "1 + 1", 2)
            == result
        )

    def test_run(self, pandasai, llm):
        df = pd.DataFrame()
        llm._output = "1"
        assert pandasai.run(df, "What number comes before 2?") == "1"

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

    def test_run_with_verbose(self, pandasai, llm):
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

    def test_run_code(self, llm):
        df = pd.DataFrame()
        pandasai = PandasAI(llm=llm)
        assert pandasai.run_code("1 + 1", df) == 2

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
        assert pandasai.conversational_answer("How much does 1 + 1 do?", "", 2) == 2
        llm.call.assert_not_called()

    def test_conversational_answer_without_privacy_enforcement(self, pandasai, llm):
        pandasai._enforce_privacy = False
        llm.call = Mock(return_value="The answer is 2")
        assert (
            pandasai.conversational_answer("How much does 1 + 1 do?", "", 2)
            == "The answer is 2"
        )
        llm.call.assert_called()

    def test_run_with_privacy_enforcement(self, pandasai, llm):
        df = pd.DataFrame({"country": ["United States", "United Kingdom", "France"]})
        pandasai._enforce_privacy = True
        pandasai._is_conversational_answer = True

        expected_prompt = f"""
Today is {date.today()}.
You are provided with a pandas dataframe (df) with 3 rows and 1 columns.
This is the result of `print(df.head(0))`:
Empty DataFrame
Columns: [country]
Index: [].

When asked about the data, your response should include a python code that describes the dataframe `df`.
Return the python code (do not import anything) and make sure to prefix the requested python code with <startCode> exactly and suffix the code with <endCode> exactly to get the answer to the following question:
How many countries are in the dataframe?

Code:
"""
        pandasai.run(df, "How many countries are in the dataframe?")
        assert pandasai._llm.last_prompt == expected_prompt

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
"""
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
This is the result of `print(df.head(5))`:
          country
0   United States
1  United Kingdom
2          France.

When asked about the data, your response should include a python code that describes the dataframe `df`.
Return the python code (do not import anything) and make sure to prefix the requested python code with <startCode> exactly and suffix the code with <endCode> exactly to get the answer to the following question:
How many countries are in the dataframe?

Code:
"""
        pandasai.run(df, "How many countries are in the dataframe?", anonymize_df=False)
        assert pandasai._llm.last_prompt == expected_prompt

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

        code = """```python<startCode>
result = {'happiness': 0.3, 'gdp': 5.5}<endCode>```"""
        assert (
            pandasai._llm._extract_code(code)
            == "result = {'happiness': 0.3, 'gdp': 5.5}"
        )

        code = """<startCode>```python
result = {'happiness': 0.49, 'gdp': 25.5}```<endCode>"""
        assert (
            pandasai._llm._extract_code(code)
            == "result = {'happiness': 0.49, 'gdp': 25.5}"
        )
        code = """<startCode>```python
result = {'happiness': 0.49, 'gdp': 25.5}```"""
        assert (
            pandasai._llm._extract_code(code)
            == "result = {'happiness': 0.49, 'gdp': 25.5}"
        )

    def test_remove_unsafe_imports(self, pandasai):
        malicious_code = """
import os
print(os.listdir())
"""
        pandasai._llm._output = malicious_code
        assert pandasai.remove_unsafe_imports(malicious_code) == "print(os.listdir())"
        with pytest.raises(Exception):
            pandasai.run_code(malicious_code, pd.DataFrame())
