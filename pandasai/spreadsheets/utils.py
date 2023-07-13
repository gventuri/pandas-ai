import requests
from bs4 import BeautifulSoup
import pandas as pd
import re


def get_google_sheet(src) -> list:
    """
    Returns a 2D array of the contents of the Google Sheet at the given URL

    Args:
        src (str): The URL of the Google Sheet

    Returns:
        grid (list): A 2D array of the contents of the Google Sheet
    """

    # The size of the Google sheet that can be read is limited
    raw_html = requests.get(src).text

    soup = BeautifulSoup(raw_html, "html.parser")
    table = soup.find("tbody")
    rows = table.find_all("tr")
    grid = []
    for row in rows:
        cols = row.find_all("td")
        clean_row = []
        for col in cols:
            clean_row.append(col.text)
        grid.append(clean_row)
    return grid


def sheet_to_df(sheet) -> list:
    """
    Returns a list of dataframes for each data table in a given spreadsheet

    Args:
        sheet (list): A 2D array of the contents of the Google Sheet

    Returns:
        dfs (list): A list of dataframes from the Google Sheet
    """

    # A dataframe starts when a header is found
    # A header is a the first instance of a set of contiguous alphanumeric columns
    # A dataframe ends when a blank row is found or an empty column is found

    num = 0  # The number of the dataframe
    headers = []  # Each header is a tuple (num, row, col_start, col_end)
    binding_headers = []
    dfs = []  # Each df is a tuple (num, df)

    # First pass: get all the headers
    for row in range(len(sheet)):
        # if every cell in the row is empty, skip row
        if all([sheet[row][col].strip() == "" for col in range(len(sheet[row]))]):
            headers += binding_headers
            binding_headers = []
            continue

        for col in range(len(sheet[row])):
            # Check if the cell is bounded by a header
            if any(
                [col >= header[2] and col <= header[3] for header in binding_headers]
            ):
                continue

            # Check if the cell is commented out
            if sheet[row][col].strip().startswith("//"):
                continue

            if re.search("[a-zA-Z]", sheet[row][col]):
                head_start = col
                head_end = col
                while head_end < len(sheet[row]) and re.search(
                    "[a-zA-Z]", sheet[row][head_end]
                ):
                    head_end += 1
                binding_headers.append([num, row, head_start, head_end])
                num += 1
    headers += binding_headers

    # Second pass: get all the dataframes
    for header in headers:
        df = []
        for row in range(header[1], len(sheet)):
            if all(
                [sheet[row][col].strip() == "" for col in range(header[2], header[3])]
            ):
                break
            df_row = []
            for col in range(header[2], header[3]):
                df_row.append(sheet[row][col])
            df.append(df_row)
        cols = df[0]
        data = df[1:]
        df = pd.DataFrame(data, columns=cols)

        # Cast all the numeric columns to numeric types
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except ValueError:
                pass
        dfs.append(df)

    return dfs


def from_google_sheets(url) -> list:
    """
    Returns the dataframes that are in a google sheet

    Args:
        url (str): The URL of the Google Sheet
    Returns:
        dfs (list): A list of dataframes from the Google Sheet
    """

    sheet = get_google_sheet(url)
    dfs = sheet_to_df(sheet)
    return dfs
