"""
This module contains helper functions for pulling data from different sources, 
Ex. Google Sheets, Excel, etc..
"""

import pandas as pd
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build


def from_google_sheets(
    spreadsheet_id, spreadsheet_name, service_account_file
) -> pd.DataFrame:
    """
    Fetches data from a Google Sheets document and returns it as a pandas DataFrame.

    Args:
        spreadsheet_id (str): The ID of the Google Sheets document.
        spreadsheet_name (str): The name of the sheet within the document to fetch data from.

    Returns:
        pd.DataFrame: DataFrame containing the data from the Google Sheets document.
    """

    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

    creds = Credentials.from_service_account_file(service_account_file, scopes=scopes)

    service = build("sheets", "v4", credentials=creds)

    # pylint: disable=no-member
    sheet = service.spreadsheets()
    result = (
        sheet.values()
        .get(spreadsheetId=spreadsheet_id, range=spreadsheet_name)
        .execute()
    )

    values = result.get("values", [])

    df_example = pd.DataFrame(values[1:], columns=values[0])

    for col in df_example.columns:
        df_example[col] = pd.to_numeric(df_example[col], errors="ignore")

    return df_example
