import pandas as pd
import matplotlib.pyplot as plt
from src.MinMaxScaler import MinMaxScaler
from src.Compute import Compute
import sys as sys

if __name__ == "__main__":
    try:
        assert len(sys.argv) == 3, "2 arguments required"
        first_course = sys.argv[1]
        second_course = sys.argv[2]
        df = pd.read_csv('./datasets/dataset_train.csv')
        for course in [first_course, second_course]:
            if course not in df.columns:
                raise ValueError(f"'{course}' is not a valid course")
        numerical_columns = df.columns[6:]
        normalized_data = MinMaxScaler.fit_transform(df[numerical_columns])
        df[numerical_columns] = normalized_data

        x = df[first_course]
        y = df[second_course]

        color_map = {
            'Ravenclaw': 'blue',
            'Slytherin': 'green',
            'Gryffindor': 'red',
            'Hufflepuff': 'yellow'
        }
        colors = df['Hogwarts House'].map(color_map)
        for house, color in color_map.items():
            plt.scatter([], [], color=color, label=house)
        plt.scatter(x, y, c=colors)
        plt.xlabel(first_course)
        plt.ylabel(second_course)
        plt.title(f"Scatter Plot: {first_course} vs {second_course}")
        plt.legend()
        plt.show()
    except Exception as error:
        print(f"error: {error}")