"""Example of using PandasAI with GoogleSheets"""

import pandas as pd
from data.sample_dataframe import dataframe

from pandasai import PandasAI
from pandasai.helpers.datasource import from_google_sheets
from pandasai.llm.openai import OpenAI

# instructions: https://docs.google.com/document/d/1uAmXI3e8Wm26lZQb0bYIJaXRdZi5WLJKNJHT59QMGEc/edit?usp=sharing


df = from_google_sheets(
    "1IRGJFUpCR0-9tPDk2MgPjQCtNft_4a8B1fHgYK7_LM8",
    "second",
    "../service_account_file.json",
)

llm = OpenAI()
pandas_ai = PandasAI(llm, verbose=True, conversational=False)

response = pandas_ai.run(
    df,
    "Calculate the sum of the gdp of north american countries",
)
print(response)
# Output: 20901884461056
