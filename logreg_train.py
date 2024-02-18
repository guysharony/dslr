import sys as sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_process import data_process
from src.data_process import data_spliter
from src.file_management import save_parameters_to_file

from src.LogisticRegression import LogisticRegression

def main():
    """
     Reads the dataset from a CSV file, preprocesses the data,
     splits it into training and test sets, trains a logistic regression model on the training data,
     evaluates its accuracy on the test data, and saves the trained model parameters to a file.
    """
    try:
        assert len(sys.argv) == 2, "1 argument required"

        dataset = pd.read_csv(sys.argv[1])
        x, y = data_process(dataset, 'train model')
        x_train, x_test, y_train, y_test = data_spliter(x, y, 0.8)

        # model
        model = LogisticRegression()

        # Training
        weights, bias = model.fit(x_train, y_train)

        # Prediction
        y_house_predictions = model.predict(x_test)

        # Display accuracy
        y_house_accuracy = np.mean(y_house_predictions == y_test)
        print(f"Predictions: {y_house_predictions.flatten()}")
        print(f"Accuracy: {y_house_accuracy * 100:.2f}%")

        # Saving thetas
        save_parameters_to_file({
            'weights': weights,
            'bias': bias
        }, 'weights')
        
        plt.show()

    except Exception as error:
        print(f"error: {error}")

if __name__ == "__main__":
    main()