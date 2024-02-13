from pandas import DataFrame

def fit_transform(data: 'DataFrame') -> 'DataFrame':
    """
    Normalize the input DataFrame by scaling each feature to a range between 0 and 1.

    Args:
        data (DataFrame): input DataFrame containing numerical features to be normalized.

    Raises:
        ValueError: if input data is not a dataframe
        ValueError: if it contains non-numerical data

    Returns:
        DataFrame: DataFrame where each feature has been scaled.
    """
    if not isinstance(data, DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")
    if not data.select_dtypes(include='number').columns.equals(data.columns):
        raise ValueError("Input DataFrame must contain only numerical features.")
    min_val = data.min()
    max_val = data.max()
    return (data - min_val) / (max_val - min_val)