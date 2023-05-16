"""
This module contains helper functions for generating plots.
"""


def is_code_for_plots(code: str,plt_code: str, plot_keywords: list[str]) -> bool:
    """
    Checks whether the code is for generating the plots.

    Returns:
        bool: True if the code is for generating the plots, False otherwise.
    """
    no_plt = code.count(plt_code) < 1
    return any(keyword_generator(code, plot_keywords)) and no_plt


def keyword_generator(code: str, plot_keywords: list[str]):
    """
     A generator function that iterates over plot_keywords.

    Returns:
        Generator
    """
    for plot_keyword in plot_keywords:
        yield plot_keyword in code
