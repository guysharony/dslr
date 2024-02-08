import sys as sys
import pandas as pd
from src.file_management import load_parameters_from_file
from logreg_train import preprocess_data
from src.LogisticRegression import LogisticRegression
from sklearn.preprocessing import LabelEncoder

house_names = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']

if __name__ == "__main__":
    try:
        assert len(sys.argv) == 3, "2 arguments required"
        df = pd.read_csv(sys.argv[1])
        # print(df)

        df = preprocess_data(df, 'test model')
        print(df.shape)
        parameters = load_parameters_from_file(sys.argv[2])
        weights = parameters['weights']
        bias = parameters['bias']
        print(weights, bias)


        model = LogisticRegression(None, None, weights, bias)
        X = df
        print(X.shape)
        predictions = model.predict(X)
        print(predictions)
        decoded_predictions = [house_names[label] for label in predictions]
        print(decoded_predictions)
    except Exception as error:
        print(f"error: {error}")
