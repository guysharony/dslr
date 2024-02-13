import sys as sys
import pandas as pd

from src.data_process import data_process
from src.LogisticRegression import LogisticRegression
from src.file_management import load_parameters_from_file, create_output_csv

house_names = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']

if __name__ == "__main__":
    try:
        assert len(sys.argv) == 3, "2 arguments required"

        df = pd.read_csv(sys.argv[1])

        x, _ = data_process(df, 'test model')

        parameters = load_parameters_from_file(sys.argv[2])
        weights = parameters['weights']
        bias = parameters['bias']

        model = LogisticRegression(None, None, weights, bias)
        predictions = model.predict(x)

        decoded_predictions = [house_names[label] for label in predictions]

        create_output_csv('houses', decoded_predictions)        

    except Exception as error:
        print(f"error: {error}")