#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script contains all the functions related to the cleaning process."""

import pandas as pd


def feature_engineering(df_cleaned):
    """
    Feature engineering.

    Return df_fe after feature engineering
    See further details in documentation.
    """
    df_cleaned = pd.concat([df_cleaned, pd.get_dummies(df_cleaned["color"])],
                           axis=1)

    # Creating 3 groups of director_name :
    # First group : only 1 film
    # Second group : Between 1 and 5 films
    # Third group : 5 films or more
    vc_director = df_cleaned["director_name"].value_counts()
    group1 = vc_director[vc_director == 1].index.values
    group2 = vc_director[(1 < vc_director) & (vc_director < 5)].index.values
    group3 = vc_director[vc_director >= 5].index.values
    i = 0
    df_cleaned['director_firstclass'] = 0
    df_cleaned['director_secondclass'] = 0
    df_cleaned['director_thirdclass'] = 0
    df_cleaned['director_unknown'] = 0
    for element in df_cleaned['director_name']:
        if element in group1:
            df_cleaned.loc[i, 'director_thirdclass'] = 1
        elif element in group2:
            df_cleaned.loc[i, 'director_secondclass'] = 1
        elif element in group3:
            df_cleaned.loc[i, 'director_firstclass'] = 1
        else:
            df_cleaned.loc[i, 'director_unknown'] = 1
        i += 1

    df_fe = df_cleaned.drop(df_cleaned.select_dtypes(
                            include=['object']).columns, axis=1)
    return df_fe
