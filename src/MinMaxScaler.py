from src.Compute import Compute

class MinMaxScaler:
    @staticmethod
    def fit_transform(data: 'pandas.core.frame.DataFrame') -> 'pandas.core.frame.DataFrame':
        min_val = data.min()
        max_val = data.max()
        # min_val = Compute.min(data)
        # max_val = Compute.max(data)

        scaled_data = (data - min_val) / (max_val - min_val)
        return scaled_data