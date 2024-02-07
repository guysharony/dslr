import pandas as pd
import sys as sys
import time
from src.MinMaxScaler import MinMaxScaler
from src.LogisticRegression import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score
# best hand convert to 0 and 1 
# scale date


def preprocess_data(df):
    def convert_to_timestamp(x):
        """Convert date objects to integers"""
        return time.mktime(x.timetuple())
    def normalize(data):
        return MinMaxScaler.fit_transform(data)

    # drop irrelevant columns
    df = df.drop(columns=['First Name', 'Last Name'])
    # replace NaN values by mean
    # df.fillna(df.mean(),inplace=True)
    df = df.dropna()
    # convert string to int
    df['Best Hand'] = df['Best Hand'].apply(lambda x : 1 if x == "Right" else 0)
    # convert date string to date object
    df['Birthday'] = pd.to_datetime(df['Birthday'])
    # convert date object to int
    df['Birthday'] = df['Birthday'].apply(convert_to_timestamp)
    # df = df.drop(columns=['Birthday'])
    # normalize numerical values
    numerical_columns = df.select_dtypes(include=['float64']).columns
    df[numerical_columns] = normalize(df[numerical_columns])
    # encode Hogwart Houses labels
    label_encoder = LabelEncoder()
    df['Hogwarts House'] = label_encoder.fit_transform(df['Hogwarts House'])
    return df

if __name__ == "__main__":
    try:
        assert len(sys.argv) == 2, "1 argument required"
        df = pd.read_csv(sys.argv[1])
        df = preprocess_data(df)

        print(df)
        # to get types 
        print(pd.DataFrame(df.dtypes).rename(columns = {0:'dtype'}))

        X =  df.drop(columns=['Index', 'Hogwarts House']).values
        y = df['Hogwarts House'].values

        model = LogisticRegression()
        model.fit(X, y)
    except Exception as error:
        print(f"error: {error}")
