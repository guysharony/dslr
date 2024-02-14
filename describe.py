import sys
import pandas as pd

from src.Compute import Compute

def main():
    """
    Read a CSV file by the 1st command line argument and compute statistics for numerical columns.
    The results are displayed in a formatted dataframe

    Raises:
        Exception: if error occurs during file reading or computation.

    """
    try:
        filename = sys.argv[1]
        dataset = pd.read_csv(filename, index_col=0)

        columns = dataset.dropna(axis=1, how='all').select_dtypes(include="number").columns.tolist()

        dataframe = pd.DataFrame(
            columns=columns,
            index=['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max', 'Median', 'Skewness', 'Kurtosis']
        )

        for column in columns:
            values = dataset[column].tolist()
            dataframe[column]['Count'] = '{:9f}'.format(Compute.count(values))
            dataframe[column]['Mean'] = '{:9f}'.format(Compute.mean(values))
            dataframe[column]['Std'] = '{:9f}'.format(Compute.std(values))
            dataframe[column]['Min'] = '{:9f}'.format(Compute.min(values))
            dataframe[column]['25%'] = '{:9f}'.format(Compute.percentile(values, 25))
            dataframe[column]['50%'] = '{:9f}'.format(Compute.percentile(values, 50))
            dataframe[column]['75%'] = '{:9f}'.format(Compute.percentile(values, 75))
            dataframe[column]['Max'] = '{:9f}'.format(Compute.max(values))
            dataframe[column]['Median'] = '{:9f}'.format(Compute.median(values))
            dataframe[column]['Skewness'] = '{:9f}'.format(Compute.skewness(values))
            dataframe[column]['Kurtosis'] = '{:9f}'.format(Compute.kurtosis(values))

        print(dataframe)

    except Exception as err:
        print(f'error: {err}')

if __name__ == "__main__":
    main()