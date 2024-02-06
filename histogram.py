import pandas as pd
import matplotlib.pyplot as plt
from src.MinMaxScaler import MinMaxScaler


if __name__ == "__main__":
    try:
        df = pd.read_csv('./datasets/dataset_train.csv')
        houses = df['Hogwarts House'].unique()
        numerical_columns = df.columns[6:]

        normalized_data = MinMaxScaler.fit_transform(df[numerical_columns])
        df[numerical_columns] = normalized_data

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