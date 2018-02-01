#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Command line dssp7_run_fit script.

This script creates the command line which can take a date entry (optional).
It generates the model from all raw data before the given date.
"""

# Import from the standard library
import argparse
import datetime
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

# Import from dssp7
from dssp7.annex.functions import get_last_raw_dataset
from dssp7.modelisation.cleaning import clean_data
from dssp7.modelisation.feature_engineering import feature_engineering
from dssp7.modelisation.models_train import fit_model

if __name__ == '__main__':
    # Command line generation
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--datetime',
        help=('generates the model from all raw data before the given date')
    )
    parser.add_argument(
        '-pi',
        '--path_input',
        help=(
            'specify the path for the input data folder' +
            ', by default the path is : ../../data/input/'
            )
    )
    parser.add_argument(
        '-pm',
        '--path_models',
        help=(
            'specify the path for the models folder' +
            ', by default the path is : ../../models/'
            )
        )
    parser.add_argument(
        '-po',
        '--path_output',
        help=(
            'specify the path for the output data folder' +
            ', by default the path is : ../../data/output/'
            )
        )
    args = parser.parse_args()

    if args.path_input is not None:
        input_path = args.path_input
    else:
        input_path = os.path.dirname(os.path.abspath(__file__)) + "/../../data/input/"

    if args.path_models is not None:
        models_path = args.path_models
    else:
        models_path = os.path.dirname(os.path.abspath(__file__)) + "/../../models/"

    if args.path_output is not None:
        output_path = args.path_output
    else:
        output_path = os.path.dirname(os.path.abspath(__file__)) + "/../../data/output/"

    # if no given date, take current date by default
    if args.datetime is not None:
        df = get_last_raw_dataset(
            input_path,
            datetime.datetime.strptime(args.datetime, '%Y%m%d')
        )
    else:
        df = get_last_raw_dataset(input_path, datetime.date.today())
    raw_name = 'raw'

    # Cleaning data
    print('Cleaning data...')
    df_cleaned = clean_data(df)
    df_cleaned.to_csv(
        output_path + 'cleaning/' + raw_name + '_cleaning.csv', index=False
    )
    print('Now Proceeding with feature engineering...')

    # Feature Engineering
    df_fe = feature_engineering(df_cleaned)
    df_fe.to_csv(
        output_path + 'feature_engineering/' + raw_name + '_fe.csv',
        index=False
    )
    print('Feature engineering done')

    # Model fitting
    model_file_name = fit_model(models_path, df_fe)
    print(model_file_name + " successfully created !")
