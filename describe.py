import sys
import pandas as pd

from src.Compute import Compute

def main():
    try:
        filename = sys.argv[1]
        dataset = pd.read_csv(filename, index_col=0)

        columns = dataset.dropna(axis=1, how='all').select_dtypes(include="number").columns.tolist()

        dataframe = pd.DataFrame(
            columns=columns,
            index=['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
        )

        for column in columns:
            values = dataset[column].tolist()
            dataframe[column]['Count'] = Compute.count(values)
            dataframe[column]['Mean'] = Compute.mean(values)
            dataframe[column]['Std'] = Compute.std(values)
            dataframe[column]['Min'] = Compute.min(values)
            dataframe[column]['25%'] = Compute.percentile(values, 25)
            dataframe[column]['50%'] = Compute.percentile(values, 50)
            dataframe[column]['75%'] = Compute.percentile(values, 75)
            dataframe[column]['Max'] = Compute.max(values)

        print(dataframe)

    except Exception as err:
        print(f'error: {err}')

if __name__ == "__main__":
    main()