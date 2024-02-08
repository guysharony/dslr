import sys
import pandas as pd
from src.LogisticRegression import LogisticRegression


def main():
    try:
        filename = './datasets/dataset_train.csv'
        dataset = pd.read_csv(filename, index_col=0)
        dataset = dataset.dropna(axis=1, how='all')

        numerical_columns = dataset.select_dtypes(include="number")
        numerical_features = numerical_columns.shape[1]

        print(numerical_columns)


    except Exception as error:
        print(f"error: {error}")

if __name__ == "__main__":
    main()