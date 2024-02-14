import pandas as pd
import matplotlib.pyplot as plt

def main():
    """
    Generate a grid of histograms and scatter plots to visualize relationships between numerical features.

    Reads a dataset from a CSV file, drops columns with all NaN values, and generates a grid of histograms
    and scatter plots to visualize relationships between numerical features for different Hogwarts houses.

    Raises:
        Exception: An error occurred during file reading or plotting.
    """
    try:
        filename = './datasets/dataset_train.csv'
        dataset = pd.read_csv(filename, index_col=0)
        dataset = dataset.dropna(axis=1, how='all')

        numerical_columns = dataset.select_dtypes(include="number")
        numerical_features = numerical_columns.shape[1]
        houses = dataset['Hogwarts House'].unique()

        fig, axs = plt.subplots(
            nrows=numerical_features,
            ncols=numerical_features,
            figsize=(
                15,
                10
            ),
            tight_layout=True
        )

        color_map = {
            'Ravenclaw': 'blue',
            'Slytherin': 'green',
            'Gryffindor': 'red',
            'Hufflepuff': 'yellow'
        }

        for i in range(numerical_features):
            for j in range(numerical_features):
                for house in houses:
                    if i == j:
                        category_values = dataset[dataset['Hogwarts House'] == house][numerical_columns.columns[i]]
                        axs[i, j].hist(category_values, color=color_map[house], alpha=0.5, density=True)
                    else:
                        category_values_x = dataset[dataset['Hogwarts House'] == house][numerical_columns.columns[j]]
                        category_values_y = dataset[dataset['Hogwarts House'] == house][numerical_columns.columns[i]]
                        axs[i, j].scatter(category_values_x, category_values_y, color=color_map[house], s=2)

                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])

                if j == 0:
                    ylabel = numerical_columns.columns[i]
                    ylabel_short = ylabel[:4]
                    axs[i, j].set_ylabel(
                        f'{ylabel_short}.' if len(ylabel) > len(ylabel_short) else f'{ylabel_short}',
                        fontsize=10
                    )

                if i == numerical_features - 1:
                    xlabel = numerical_columns.columns[j]
                    xlabel_short = xlabel[:4]
                    axs[i, j].set_xlabel(
                        f'{xlabel_short}.' if len(xlabel) > len(xlabel_short) else f'{xlabel_short}',
                        fontsize=10
                    )

        fig.legend(houses, loc='lower right')
        plt.show()
    except Exception as err:
        print(f'error: {err}')

if __name__ == "__main__":
    main()