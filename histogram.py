import pandas as pd
import matplotlib.pyplot as plt
from src.MinMaxScaler import MinMaxScaler
from src.Compute import Compute


if __name__ == "__main__":
    try:
        df = pd.read_csv('./datasets/dataset_train.csv')
        houses = df['Hogwarts House'].unique()
        numerical_columns = df.columns[6:]
        normalized_data = MinMaxScaler.fit_transform(df[numerical_columns])
        df[numerical_columns] = normalized_data

        # calculate how scores spread out for each course
        courses_std = []
        for i, column in enumerate(df.columns[6:]):
            courses_std.append(Compute.std(df[column].dropna()))
            print(f"{column}: {courses_std[i]}")
        min_std_index = courses_std.index(Compute.min(courses_std))
        print(f"{df.columns[6:][min_std_index]} has the most homogeneous score distribution between all four houses.")

        # plot histograms 
        fig, axs = plt.subplots(nrows=2, ncols=7, figsize=(25, 6))
        axs = axs.flatten()

        for i, column in enumerate(df.columns[6:]):
            for house in houses:
                category_values = df[df['Hogwarts House'] == house][column]
                axs[i].hist(category_values, alpha=0.5, density=True)
            axs[i].set_title(f'{column}')
            axs[i].set_xlabel('Score')
            axs[i].set_ylabel('Frequency')

        fig.delaxes(axs[13])
        fig.legend(houses, loc='lower right')
        plt.tight_layout()
        plt.show()
    except Exception as error:
        print(f"error: {error}")