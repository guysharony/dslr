import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data_process import data_process
from src.data_process import data_spliter
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

        gradient_descent = ['full batch', 'stochastic', 'mini-batch']

        accuracy_results = []
        time_results = []

        batch_size = [None, 1, 32]
        plt.figure(figsize=(15, 5))

        for i, algorithm in enumerate(gradient_descent):
            print(f"{algorithm} gradient descent:")

            # model
            model = LogisticRegression(batch_size=batch_size[i])

            plt.subplot(1, 3, i + 1)
            plt.title(f"{algorithm} gradient descent")

            start_time = time.time()

            # Training
            model.fit(x_train, y_train)

            end_time = time.time()
            execution_time = end_time - start_time

            # Prediction
            y_house_predictions = model.predict(x_test)

            # Compute accuracy
            accuracy = model.accuracy(y_house_predictions, y_test)
            accuracy_results.append(accuracy)
            time_results.append(execution_time)

            print(f"Optimization algorithm: {algorithm}")
            print(f"Accuracy: {accuracy * 100:.2f}%")
            print(f"Execution time: {execution_time:.2f} seconds")
            if i < len(gradient_descent) - 1:
                print()

        plt.tight_layout()
        plt.show()

    except Exception as error:
        print(f"error: {error}")

if __name__ == "__main__":
    main()