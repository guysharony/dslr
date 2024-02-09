import time
import sys as sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.MinMaxScaler import MinMaxScaler

def convert_to_timestamp(x):
    """Convert date objects to integers"""
    return time.mktime(pd.to_datetime(x).timetuple())

def normalize(data):
    return MinMaxScaler.fit_transform(data)

def data_process(dataset, status):
    dataset = dataset.dropna(axis=1, how='all')
    dataset = dataset.drop(columns=['First Name', 'Last Name'])
    dataset = dataset.dropna()

    dataset['Best Hand'] = dataset['Best Hand'].apply(lambda x : 1 if x == "Right" else 0)
    dataset['Birthday'] = dataset['Birthday'].apply(convert_to_timestamp)

    numerical_columns = dataset.select_dtypes(include=['float64']).columns
    dataset[numerical_columns] = normalize(dataset[numerical_columns])

    if status == 'train model':
        label_encoder = LabelEncoder()
        dataset['Hogwarts House'] = label_encoder.fit_transform(dataset['Hogwarts House'])

    x = dataset.drop(
        columns=[
            column for column in ['Index', 'Hogwarts House'] if column in dataset.columns
        ]
    ).values
    y = dataset['Hogwarts House'].values if 'Hogwarts House' in dataset.columns else []

    return x, y
