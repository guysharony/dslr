import numpy as np
import pandas as pd
from datetime import datetime

from src.MinMaxScaler import MinMaxScaler
from src.LogisticRegression import LogisticRegression

def convert_birthday(date):
    date_object = datetime.strptime(date, '%Y-%m-%d')
    epoch = datetime(1970, 1, 1)

    return (date_object - epoch).days

def convert_house(house):
    houses = {
        'Ravenclaw': 0,
        'Slytherin': 1,
        'Gryffindor': 2,
        'Hufflepuff': 3
    }

    return houses[house]

def normalizing(dataset):
    numerical_columns = dataset.select_dtypes(include=['float64']).columns
    dataset[numerical_columns] = MinMaxScaler.fit_transform(dataset[numerical_columns])

    return dataset

def pre_process_data(dataset):
    dataset['Best Hand'] = dataset['Best Hand'].apply(lambda x : 1 if x == "Right" else 0)
    dataset['Birthday'] = dataset['Birthday'].apply(convert_birthday)

    dataset = normalizing(dataset)
    dataset['Hogwarts House'] = dataset['Hogwarts House'].apply(convert_house)

    x = dataset.drop(columns=['Hogwarts House']).values
    y = dataset['Hogwarts House'].values

    return x, y

def main():
    try:
        filename = './datasets/dataset_train.csv'
        dataset = pd.read_csv(filename, index_col=0)
        dataset = dataset.dropna(axis=1, how='all')
        dataset = dataset.drop(columns=['First Name', 'Last Name'])
        dataset = dataset.dropna()

        # Preprocessing dataset
        x, y = pre_process_data(dataset)

        # Model
        models = []
        for i in range(4):
            y_train_class = (y == i).astype(int)

            model = LogisticRegression(
                thetas=np.zeros((x.shape[1] + 1)),
                alpha=0.001,
                max_iter=1_000_000
            )
            model.fit_(x, y_train_class)

            models.append(model)

        y_citizens_probabilities = np.hstack([model.predict_(x) for model in models])
        y_citizens_predictions = np.argmax(y_citizens_probabilities, axis=1).reshape(-1, 1)

        y_citizens_accuracy = np.mean(y_citizens_predictions == y)

        print(f"Predictions: {y_citizens_predictions.flatten()}")
        print(f"Accuracy: {y_citizens_accuracy * 100}%")

    except Exception as error:
        print(f"error: {error}")

if __name__ == "__main__":
    main()