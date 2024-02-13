import numpy as np
from typing import Union

class Compute:

    @staticmethod
    def numeric(values) -> list[float]:
        return [float(value) for value in values if not np.isnan(float(value))]

    @staticmethod
    def sum(values) -> Union[float, None]:
        numeric_values = Compute.numeric(values)

        _sum = 0
        for value in numeric_values:
            _sum += value

        return _sum

    @staticmethod
    def count(values) -> Union[float, None]:
        if len(values) == 0:
            return None

        return sum(1 for value in values if not np.isnan(float(value)))

    @staticmethod
    def mean(values) -> Union[float, None]:
        _sum = Compute.sum(values)
        _len = Compute.count(values)

        if not _sum or not _len:
            return None

        return _sum / _len

    @staticmethod
    def percentile(values, p) -> Union[float, None]:
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
        variance = Compute.var(values)

        if variance is None:
            return None

        return variance ** 0.5

    @staticmethod
    def median(values) -> Union[float, None]:
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