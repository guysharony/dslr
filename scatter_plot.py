import sys
import pandas as pd
import matplotlib.pyplot as plt

from pandas.api.types import is_numeric_dtype
from src.min_max_scaler import fit_transform

def main():
    """
    Plot a scatter plot comparing two courses' scores for students from different Hogwarts houses.
    Reads a dataset from a CSV file and plots a scatter plot comparing the scores of two specified courses 
    for students from different Hogwarts houses.

    Raises:
        AssertionError: Raised when incorrect number of command-line arguments is provided.
        ValueError: Raised when the same course is specified for both axes or when the specified course does not exist in the dataset.
        TypeError: Raised when the specified course is not a valid numerical course.
        Exception: Raised for any other errors that occur during execution.

    """
    try:
        assert len(sys.argv) == 3, "2 arguments required"
        first_course = sys.argv[1]
        second_course = sys.argv[2]
        if first_course == second_course:
            raise ValueError("Cannot plot self-comparison for one course, choose two different courses")

        dataset = pd.read_csv('./datasets/dataset_train.csv')
        for course in [first_course, second_course]:
            if course not in dataset.columns:
                raise ValueError(f"course '{course}' does not exist")
            if is_numeric_dtype(dataset[course]) == False:
                raise TypeError(f"{course} is not a valid course")

        # Normalizing dataset
        numerical_columns = dataset.columns[6:]
        normalized_data = fit_transform(dataset[numerical_columns])
        dataset[numerical_columns] = normalized_data

        x = dataset[first_course]
        y = dataset[second_course]

        color_map = {
            'Ravenclaw': 'blue',
            'Slytherin': 'green',
            'Gryffindor': 'red',
            'Hufflepuff': 'yellow'
        }
        colors = dataset['Hogwarts House'].map(color_map)

        # Displaying graph
        for house, color in color_map.items():
            plt.scatter([], [], color=color, label=house)
        plt.scatter(x, y, c=colors)
        plt.xlabel(first_course)
        plt.ylabel(second_course)
        plt.title(f"Scatter Plot: {first_course} vs {second_course}")
        plt.legend(loc='lower right', bbox_to_anchor=(1.05, 0), borderaxespad=0)
        plt.show()

    except Exception as error:
        print(f"error: {error}")

if __name__ == "__main__":
    main()