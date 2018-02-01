#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script contains all the functions related to the cleaning process."""

import pandas as pd


def fill_na_median(df, list_column_median):
    """Apply median function to the list of df's column given."""
    for i in list_column_median:
        df.update(df[i].fillna(df[i].median()))


def clean_data(df):
    """
    Cleaning data from df.

    Return df after data cleaning
    See further details in documentation.
    """
    list_column_median = [
        "num_critic_for_reviews",
        "duration",
        "final_budget",
    ]

    fill_na_median(df, list_column_median)

    df.update(df["color"].fillna(" Black and White"))

    df_cleaned = df[pd.notnull(df['director_name'])]
    df_cleaned.reset_index(inplace=True, drop=True)

    return df_cleaned
