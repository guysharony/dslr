import pandas as pd
import sys as sys
import time
from src.MinMaxScaler import MinMaxScaler
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
    # convert string to int
    df['Best Hand'] = df['Best Hand'].apply(lambda x : 1 if x == "Right" else 0)
    # convert date string to date object
    df['Birthday'] = pd.to_datetime(df['Birthday'])
    # convert date object to int
    df['Birthday'] = df['Birthday'].apply(convert_to_timestamp)
    # normalize numerical values
    numerical_columns = df.select_dtypes(include=['float64']).columns
    df[numerical_columns] = normalize(df[numerical_columns])
    return df

if __name__ == "__main__":
    try:
        assert len(sys.argv) == 2, "1 argument required"
        df = pd.read_csv(sys.argv[1])
        df = preprocess_data(df)
        
        print(df)
        # to get types 
        print(pd.DataFrame(df.dtypes).rename(columns = {0:'dtype'}))


    except Exception as error:
        print(f"error: {error}")
