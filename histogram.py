import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    df = pd.read_csv('./datasets/dataset_train.csv')
    # print(df.shape)
    # print(df)
    # df_sorted = df.sort_values(by='Hogwarts House')
    # print(df_sorted)
    gryffindor = df[df['Hogwarts House'] == 'Gryffindor']['Arithmancy']
    hufflepuff = df[df['Hogwarts House'] == 'Hufflepuff']['Arithmancy']
    ravenclaw = df[df['Hogwarts House'] == 'Ravenclaw']['Arithmancy']
    slytherin = df[df['Hogwarts House'] == 'Slytherin']['Arithmancy']

    plt.hist(gryffindor, alpha=0.5, label='Gryffindor')
    plt.hist(hufflepuff, alpha=0.5, label='Hufflepuff')
    plt.hist(ravenclaw, alpha=0.5, label='Ravenclaw')
    plt.hist(slytherin, alpha=0.5, label='Slythrin')

    plt.legend()
    plt.show()
    