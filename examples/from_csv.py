"""Example of using PandasAI with a CSV file."""
import os

import pandas as pd
from llm_client import LLMAPIClientType, init_sync_llm_api_client

from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
from pandasai.llm.client_sdk import ClientSDK

df = pd.read_csv("examples/data/Loan payments data.csv")

llm = OpenAI()
pandas_ai = PandasAI(llm, verbose=True, conversational=True)
response = pandas_ai(df, "How many loans are from men and have been paid off?")
print(response)
# Output: 247 loans have been paid off by men.

# With ClientSDK

llm_client = init_sync_llm_api_client(
    LLMAPIClientType.OPEN_AI, api_key=os.environ["OPENAI_API_KEY"]
)
client_sdk = ClientSDK(
    llm_client,
    frequency_penalty=0,
    max_tokens=512,
    model="text-davinci-003",
    presence_penalty=0.6,
    temperature=0,
    top_p=1,
)
pandas_ai = PandasAI(client_sdk, verbose=True)
response = pandas_ai(df, "How many loans are from men and have been paid off?")
print(response)
