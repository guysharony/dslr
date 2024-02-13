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

def data_spliter(x, y, proportion):
    """Shuffles and splits the dataset (given by x and y) into a training and a test set,
        while respecting the given proportion of examples to be kept in the training set.
    Args:
        x: has to be an numpy.array, a matrix of dimension m * n.
        y: has to be an numpy.array, a vector of dimension m * 1.
        proportion: has to be a float, the proportion of the dataset that will be assigned to the
            training set.
    Return:
        (x_train, x_test, y_train, y_test) as a tuple of numpy.array
        None if x or y is an empty numpy.array.
        None if x and y do not share compatible dimensions.
        None if x, y or proportion is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    if x.__class__ != np.ndarray or y.__class__ != np.ndarray or proportion.__class__ != float:
        return None

    if x.size == 0 or y.size == 0:
        return None

    if x.shape[0] != y.shape[0]:
        return None

    dataset = [list(i) + [j] for i, j in zip(x, y)]
    np.random.shuffle(dataset)

    x_dataset = np.array([i[:-1] for i in dataset])
    y_dataset = np.array([i[-1] for i in dataset])

    split_index = int(x.shape[0] * proportion)

    x_train, x_test = x_dataset[:split_index], x_dataset[split_index:]
    y_train, y_test = y_dataset[:split_index], y_dataset[split_index:]

    return x_train, x_test, y_train, y_test