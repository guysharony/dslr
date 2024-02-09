import pandas

class MinMaxScaler:
    @staticmethod
    def fit_transform(data: 'pandas.core.frame.DataFrame') -> 'pandas.core.frame.DataFrame':
        min_val = data.min()
        max_val = data.max()

        scaled_data = (data - min_val) / (max_val - min_val)
        return scaled_data