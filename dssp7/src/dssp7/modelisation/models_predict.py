#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script contains all the functions related to the prediction process."""


def get_predict_score(model, df_fe):
    """
    Return the prediction score.

    Predict the film score.
    Parameters:
    model - trained model from models_train.py
    df_fe - df after feature engineering
    """
    score = model.predict(df_fe.drop("imdb_score", axis=1))

    return score
