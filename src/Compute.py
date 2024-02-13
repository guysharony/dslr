import numpy as np
from typing import Union

class Compute:

    @staticmethod
    def numeric(values) -> list[float]:
        """
        Extracts numeric values from a list, filtering out NaNs

        Args:
            values (list): list of values

        Returns:
            list[float]: list of numeric values
        """
        return [float(value) for value in values if not np.isnan(float(value))]

    @staticmethod
    def sum(values) -> Union[float, None]:
        """
        Calculates the sum of the values 

        Args:
            values (list): list of values

        Returns:
            Union[float, None]: sums of the values
        """
        numeric_values = Compute.numeric(values)

        _sum = 0
        for value in numeric_values:
            _sum += value

        return _sum

    @staticmethod
    def count(values) -> Union[float, None]:
        """
        Counts the number of numeric values.

        Args:
            values (list): list of values

        Returns:
            Union[float, None]: number of numeric values
        """
        if len(values) == 0:
            return None

        return sum(1 for value in values if not np.isnan(float(value)))

    @staticmethod
    def mean(values) -> Union[float, None]:
        """
        Calculate the mean of a list of values.

        Args:
            values (values): list of values

        Returns:
            Union[float, None]: the mean of the values
        """

        _sum = Compute.sum(values)
        _len = Compute.count(values)

        if not _sum or not _len:
            return None

        return _sum / _len

    @staticmethod
    def percentile(values, p) -> Union[float, None]:
        """
        Calculate the p-th percentile of a list of values
        A percentile describes the position of a particular value within a dataset
        Args:
            values (list): list of values
            p (int): the selected percentile, should be between 0 and 100

        Returns:
            Union[float, None]: the p-th percentile of a list of values.
        """
        if len(values) == 0:
            return None

        if not 0 <= p <= 100:
            return None

        numeric_values = Compute.numeric(values)

        data_sorted = sorted(numeric_values)
        index = (p / 100) * (len(data_sorted) - 1)
        lower_index = int(index // 1)
        upper_index = lower_index + 1
        lower_value = data_sorted[lower_index]
        upper_value = data_sorted[upper_index]

        return (1 - (index - lower_index)) * lower_value + (index - lower_index) * upper_value

    @staticmethod
    def min(values) -> Union[float, None]:
        """
        Find the smallest value in a list of values.

        Args:
            values (list): list of values

        Returns:
            Union[float, None]: return the minimum value 
        """
        
        if len(values) == 0:
            return None

        _min = None
        for value in values:
            if not np.isnan(float(value)):
                if _min is None or value < _min:
                    _min = value

        return _min

    @staticmethod
    def max(values) -> Union[float, None]:
        """
        Find the largest value in a list of values.

        Args:
            values (list): list of values

        Returns:
            Union[float, None]: return the maximum value 
        """
        if len(values) == 0:
            return None

        _max = None
        for value in values:
            if not np.isnan(float(value)):
                if _max is None or value > _max:
                    _max = value

        return _max

    @staticmethod
    def var(values) -> Union[float, None]:
        """
        Calculate the variance of a list of values.
        Variance measures the average squared deviation of data points from their mean
        to quantify the spread of a dataset

        Args:
            values (list): list of values

        Returns:
            Union[float, None]: the variance of a list of values
        """

        _mean = Compute.mean(values)

        if not _mean:
            return None

        _len = Compute.count(values)
        _sum = Compute.sum([(i - _mean) ** 2 for i in values])

        if not _sum or not _len:
            return None

        return _sum / (_len - 1)

    @staticmethod
    def std(values) -> Union[float, None]:
        """
        Calculate the standard deviation of a list of values.
        Standard deviation is the square root of the variance
        and expresses the spread in the same units as the original data.

        Args:
            values (list): list of values

        Returns:
            Union[float, None]: the standard deviation of the list of values
        """
        variance = Compute.var(values)

        if variance is None:
            return None

        return variance ** 0.5

    @staticmethod
    def median(values) -> Union[float, None]:
        """
        Calculate the median of a list of values.
        The median is a measure of central tendency in a dataset,
        representing the middle value when the data points are arranged in ascending or descending order.

        Args:
            values (list): list of values

        Returns:
            Union[float, None]: the median of the list of values
        """
        numeric_values = Compute.numeric(values)

        if not numeric_values:
            return None

        sorted_values = sorted(numeric_values)
        n = len(sorted_values)
        middle_index = n // 2

        if n % 2 == 0:
            return (sorted_values[middle_index - 1] + sorted_values[middle_index]) / 2
        else:
            return sorted_values[middle_index]


    @staticmethod
    def skewness(values) -> Union[float, None]:
        """
        Calculate the skewness of a list of values.
        Skewness measures the asymmetry of the distribution of a set of data points around its mean

        Args:
            values (list): list of values

        Returns:
            Union[float, None]: the skewness of the list of values
        """
        numeric_values = Compute.numeric(values)

        if not numeric_values:
            return None

        n = len(numeric_values)
        _mean = Compute.mean(numeric_values)
        _std = Compute.std(numeric_values)

        if _mean is None:
            return None
        
        if _std is None:
            return None

        _sum = Compute.sum(((value - _mean) ** 3) / n for value in numeric_values)
        if not _sum:
            return None

        return _sum / (_std ** 3)


    @staticmethod
    def kurtosis(values) -> Union[float, None]:
        """
        Calculate the kurtosis of a list of values.
        Kurtosis quantifies the "peak" of the distribution of a dataset relative to the normal distribution

        Args:
            values (list): list of values

        Returns:
            Union[float, None]: the kurtosis of the list of values
        """
        numeric_values = Compute.numeric(values)

        if not numeric_values:
            return None

        n = len(numeric_values)
        _mean = Compute.mean(numeric_values)
        _std = Compute.std(numeric_values)

        if _mean is None:
            return None

        if _std is None:
            return None

        _sum = Compute.sum(((value - _mean) ** 4) for value in numeric_values)
        if not _sum:
            return None

        return (_sum / (n * (_std ** 4))) - 3