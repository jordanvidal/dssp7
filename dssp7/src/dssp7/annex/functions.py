#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script contains the subsidiary functions."""

import pickle
import os
import datetime
import pandas as pd


def compare(a, b):
    """Return True if a and b have the same alphanumerical characters."""
    return [c for c in a if c.isalnum()] == [c for c in b if c.isalnum()]


def get_last_model(datetime_input, path_model):
    """Return the lastest fitted model before the input date."""
    list_name_model = os.listdir(path_model)
    list_date_model = [
        date_str_to_datetime(x[len('model') + 1:]) for x in list_name_model
    ]

    list_date_model_before_datetime = [
        x for x in list_date_model if x <= datetime.datetime.combine(
            datetime_input, datetime.time(23, 59)
        )
    ]

    date_model_plus_recent = max(list_date_model_before_datetime)
    name_model = "%s_%s" % (
        'model',
        date_datetime_to_str(date_model_plus_recent)
    )
    path = path_model + name_model + '.pkl'
    pkl_file = open(path)
    model = pickle.load(pkl_file)

    return model


def get_last_raw_dataset(input_path, datetime_input):
    """Return concat form of all raw data before the input date."""
    list_name_raw = os.listdir(input_path + 'raw_data/')
    list_date_raw = [
        date_str_to_datetime(x[len('raw') + 1:]) for x in list_name_raw
    ]

    list_date_raw_before_datetime = [
        x for x in list_date_raw if x <= datetime.datetime.combine(
            datetime_input, datetime.time(23, 59)
        )
    ]
    list_raw = []
    for element in list_date_raw_before_datetime:
        name_raw = "%s_%s" % (
            'raw',
            date_datetime_to_str(element)
        )
        path = input_path + 'raw_data/' + name_raw + '.csv'
        list_raw.append(pd.read_csv(path, sep=';'))
    raw = pd.concat(list_raw)
    return raw


def date_datetime_to_str(date_dt):
    """
    Convert input date to String.

    input: datetime(2017,07,14,23,59)
    output: "2017_07_14_23_59"
    """
    return "%s_%s_%s_%s_%s" % (
        str(date_dt.year).zfill(2),
        str(date_dt.month).zfill(2),
        str(date_dt.day).zfill(2),
        str(date_dt.hour).zfill(2),
        str(date_dt.minute).zfill(2))


def date_str_to_datetime(date_str):
    """
    Convert input String to date.

    input: "2017_07_14_23_59.pkl"
    output: datetime(2017,07,14,23,59)
    """
    date_str = date_str.split(".")[0]
    date_str_split = [int(x) for x in date_str.split("_")]
    date_dt = datetime.datetime(
        date_str_split[0],
        date_str_split[1],
        date_str_split[2],
        date_str_split[3],
        date_str_split[4])
    return date_dt
