#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Command line dssp7_run_predict script.

This script creates the command line which can take a date entry (optional).
It predicts the score based on the latest generated model before given date.
"""

# Import from the standard library
import argparse
import datetime
import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

# Import from dssp7
from dssp7.annex.functions import get_last_model
from dssp7.modelisation.cleaning import clean_data
from dssp7.modelisation.feature_engineering import feature_engineering
from dssp7.modelisation.models_predict import get_predict_score

if __name__ == '__main__':
    # Command line generation
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--datetime',
                        help=(
                            'insert a datetime with format YYYYmmdd,' +
                            'return prediction of the score using the model' +
                            'fitted before this date'
                        )
                        )
    
    parser.add_argument('-pi',
                        '--path_input',
                        help=(
                            'insert data_input path, by default the path is:' +
                            '../../data/input/'
                            ))

    parser.add_argument('-po',
                        '--path_output',
                        help=(
                            'insert data_output path, by default the path is:' +
                            '../../data/output/'
                            ))

    parser.add_argument('-pm',
                        '--path_models',
                        help=(
                            'insert model path, by default the path is:' +
                            '../../models/'
                            ))

    args = parser.parse_args()

    if args.path_input is not None:
        path_data_input = args.path_input
    else:
        path_data_input = os.path.dirname(os.path.abspath(__file__)) + '/../../data/input/'

    if args.path_output is not None:
        path_data_output = args.path_output
    else:
        path_data_output = os.path.dirname(os.path.abspath(__file__)) + '/../../data/output/'   

    if args.path_models is not None:
        path_models = args.path_models
    else:
        path_models = os.path.dirname(os.path.abspath(__file__)) + '/../../models/'

    # if no given date, take current date by default
    if args.datetime is not None:
        model = get_last_model(
            datetime.datetime.strptime(args.datetime, '%Y%m%d')
            , path_models
        )
    else:
        model = get_last_model(datetime.date.today(), path_models)

    # Input pre-processing
    input_name = 'to_predict'
    df = pd.read_csv(path_data_input + 'to_predict/' + input_name + '.csv', sep=';')
    # Cleaning input data
    df_cleaned = clean_data(df)
    df_cleaned.to_csv(
        path_data_output + 'cleaning/' + input_name + '_cleaning.csv', index=False
    )
    # Feature engineering
    df_fe = feature_engineering(df_cleaned)
    df_fe.to_csv(
        path_data_output + 'feature_engineering/' + input_name + '_fe.csv', index=False
    )
    # Create output dataframe
    df_result = pd.DataFrame(
        index=range(0, df_fe.shape[0]),
        columns=['movie_title', 'predict_score', 'real_score']
    )
    df_result['movie_title'] = df_cleaned['movie_title']
    df_result['real_score'] = df_fe['imdb_score']
    df_result['predict_score'] = get_predict_score(model, df_fe)
    path = path_data_output + 'predicts/' + input_name + '_result.csv'
    df_result.to_csv(path, index=False)
    print('Prediction done, find your file in {}'.format(os.path.abspath(path)))
