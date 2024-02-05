import sys
import pandas as pd

def main():
    try:
        filename = sys.argv[1]
        dataset = pd.read_csv(filename, index_col=0)

        columns = dataset.select_dtypes(include="number").columns.tolist()

        for column in columns:
            values = dataset[column].tolist()
            print(values)

    except Exception as err:
        print(f'error: {err}')

if __name__ == "__main__":
    main()