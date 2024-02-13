import time
import sys as sys
import numpy as np
import pandas as pd

from src.min_max_scaler import fit_transform

house_names = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']

def decode_house(house: int) -> str:
    """
    Decode the encoded house index to its corresponding house name.

    Args:
        house (int): The encoded index of the house.

    Returns:
        str : The corresponding house name
    """
    return house_names[house]

def encode_house(house: str) -> int:
    """
    Encode the house name to its corresponding index.

    Args:
        house (str): The name of the house
    Returns:
        int: The encoded index of the house
    """
    return house_names.index(house)

def convert_to_timestamp(x) -> int:
    """
    Convert date objects to timestamps.

    Args:
        x: the date object

    Returns:
        int: the timestamp of the date
    """
    return time.mktime(pd.to_datetime(x).timetuple())

def data_process(dataset, status):
    """
    Preprocess the dataset by performing transformations.

    Args:
        dataset (pd.DataFrame): dataset to be processed
        status (str): indicates the purpose of processsing

    Returns:
        Tuple[np.ndarray, np.ndarray]: The processed features (x) and labels (y).
    """
    dataset = dataset.dropna(axis=1, how='all')
    dataset = dataset.drop(columns=['First Name', 'Last Name'])

    if status == 'train model':
        dataset = dataset.dropna()

    dataset['Best Hand'] = dataset['Best Hand'].apply(lambda x : 1 if x == "Right" else 0)
    dataset['Birthday'] = dataset['Birthday'].apply(convert_to_timestamp)

    numerical_columns = dataset.select_dtypes(include=['float64']).columns
    dataset[numerical_columns] = fit_transform(dataset[numerical_columns])

    if status == 'train model':
        dataset['Hogwarts House'] = dataset['Hogwarts House'].apply(encode_house)

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