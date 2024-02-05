import numpy as np
from typing import Union

class Statistician:

    @staticmethod
    def count(x) -> Union[float, None]:
        if len(x) == 0:
            return None

        return sum(1 for value in x if not np.isnan(float(value)))

    @staticmethod
    def mean(x) -> Union[float, None]:
        if len(x) == 0:
            return None

        return sum(x) / len(x)

    @staticmethod
    def median(x) -> Union[float, None]:
        return Statistician.percentile(x, 50)

    @staticmethod
    def quartile(x) -> Union[float, None]:
        if len(x) == 0:
            return None

        return [
            Statistician.percentile(x, 25),
            Statistician.percentile(x, 75)
        ]

    @staticmethod
    def percentile(x, p) -> Union[float, None]:
        if not x:
            return None

        if not 0 <= p <= 100:
            return None

        data_sorted = sorted(x)
        index = (p / 100) * (len(data_sorted) - 1)
        lower_index = int(index // 1)
        upper_index = lower_index + 1
        lower_value = data_sorted[lower_index]
        upper_value = data_sorted[upper_index]

        return (1 - (index - lower_index)) * lower_value + (index - lower_index) * upper_value

    @staticmethod
    def var(x) -> Union[float, None]:
        if len(x) == 0:
            return None

        mean = Statistician.mean(x)
        return sum([(i - mean) ** 2 for i in x]) / (len(x) - 1)

    @staticmethod
    def std(x) -> Union[float, None]:
        variance = Statistician.var(x)

        if variance is None:
            return None

        return variance ** 0.5