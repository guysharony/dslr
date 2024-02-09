import sys as sys
import pandas as pd

from src.DataProcess import data_process
from src.LogisticRegression import LogisticRegression
from src.file_management import save_parameters_to_file

if __name__ == "__main__":
    try:
        assert len(sys.argv) == 2, "1 argument required"

        dataset = pd.read_csv(sys.argv[1])
        x, y = data_process(dataset, 'train model')

        model = LogisticRegression()
        weights, bias = model.fit(x, y)

        save_parameters_to_file({
            'weights': weights,
            'bias': bias
        }, 'weights')

    except Exception as error:
        print(f"error: {error}")
