#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script contains all the functions related to the training process."""

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import datetime
import pickle


def get_model_error(model, x_test, y_test):
    """
    Return the error of the model.

    The result is the error of the test samples.
    """
    y2_predict = model.predict(x_test)
    error_test = mean_squared_error(y_test, y2_predict)

    return error_test


def fit_model(models_path, df_fe):
    """Return the name of the pickled model."""
    x_train, x_test, y_train, y_test = train_test_split(
        df_fe.drop('imdb_score', axis=1), df_fe['imdb_score'], test_size=0.4
    )

    # Another function to get the best model will be implemented soon !
    model = RandomForestRegressor(max_features=9)
    model.fit(x_train, y_train)

    error_test = get_model_error(model, x_test, y_test)
    print("\nError for test sample:")
    print(error_test)
    print("\n")

    # Saving model as pickle file
    file_folder = models_path
    model_file_name = (
        "model" +
        datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M") +
        ".pkl"
    )
    pickle.dump(model, open(file_folder + model_file_name, "wb"))

    return model_file_name
