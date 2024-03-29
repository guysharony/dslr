import sys as sys
import pandas as pd

from src.data_process import data_process
from src.data_process import decode_house
from src.file_management import create_output_csv
from src.file_management import load_parameters_from_file

from src.LogisticRegression import LogisticRegression

house_names = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']

def main():
    """
    Reads the dataset from a CSV file, preprocesses the data,
    loads the pre-trained logistic regression model parameters from a file,
    makes predictions on the dataset, decodes the predicted labels
    and saves the decoded predictions to a CSV file.
    """
    try:
        assert len(sys.argv) == 3, "2 arguments required"

        # Process dataset
        dataset = pd.read_csv(sys.argv[1])
        x, _ = data_process(dataset, 'test model')

        # Load parameters
        parameters = load_parameters_from_file(sys.argv[2])
        thetas = parameters['thetas']

        # Logistic regression
        model = LogisticRegression(None, None, thetas)
        predictions = model.predict(x)

        # Decoding predictions
        decoded_predictions = [decode_house(label) for label in predictions]

        # Saving decoded predictions
        create_output_csv('houses', decoded_predictions)
    except Exception as error:
        print(f"error: {error}")

if __name__ == "__main__":
    main()