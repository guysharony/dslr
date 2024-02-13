from pandas import DataFrame

def fit_transform(data: 'DataFrame') -> 'DataFrame':
    min_val = data.min()
    max_val = data.max()

    return (data - min_val) / (max_val - min_val)