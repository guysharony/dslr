import sys
import pandas as pd

from src.Statistician import Statistician

def main():
    try:
        filename = sys.argv[1]
        dataset = pd.read_csv(filename, index_col=0)

        columns = dataset.select_dtypes(include="number").columns.tolist()

        dataframe = pd.DataFrame(
            columns=columns,
            index=['Count', 'Mean', 'Std', 'Min', '25%', '50%','75%','Max']
        )

        for column in columns:
            values = dataset[column].tolist()
            dataframe[column]['Count'] = Statistician.count(values)
            dataframe[column]['Mean'] = Statistician.mean(values)
            dataframe[column]['Std'] = Statistician.std(values)
            dataframe[column]['Min'] = min(values)
            dataframe[column]['25%'] = Statistician.percentile(values, 25)
            dataframe[column]['50%'] = Statistician.percentile(values, 50)
            dataframe[column]['75%'] = Statistician.percentile(values, 75)
            dataframe[column]['Max'] = max(values)

        print(dataframe)

    except Exception as err:
        print(f'error: {err}')

if __name__ == "__main__":
    main()