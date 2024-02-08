import numpy as np

class LogisticRegression:
    def __init__(self, thetas, alpha=0.001, max_iter=100000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas

    @staticmethod
    def sigmoid_(x):
        if x.size == 0:
            return None

        return 1 / (1 + np.exp(-x))

    def gradient_(self, x, y):
        if x.__class__ != np.ndarray or y.__class__ != np.ndarray or self.thetas.__class__ != np.ndarray:
            return None

        if x.shape[1] + 1 != self.thetas.shape[0]:
            return None

        if x.size == 0 or y.size == 0 or self.thetas.size == 0:
            return None

        m = x.shape[0]

        x_prime = np.hstack((np.ones((m, 1)), x))
        y_hat = self.predict_(x)
        return x_prime.T.dot(y_hat - y) / m

    def predict_(self, x):
        if x.__class__ != np.ndarray or self.thetas.__class__ != np.ndarray:
            return None

        if x.size == 0 or self.thetas.size == 0:
            return None

        if x.shape[1] + 1 != self.thetas.shape[0]:
            return None

        x_prime = np.hstack((np.ones((x.shape[0], 1)), x))
        return LogisticRegression.sigmoid_(np.dot(x_prime, self.thetas))

    def loss_(self, y, y_hat):
        eps = 1e-15

        if y.__class__ != np.ndarray or y_hat.__class__ != np.ndarray:
            return None

        if y.size == 0 or y_hat.size == 0:
            return None

        m = y.shape[0]
        ones_vector = np.ones((m, 1))
        return - (1 / m) * np.sum(
            y * np.log(y_hat + eps) + (ones_vector - y) * np.log(ones_vector - y_hat + eps)
        )

    def fit_(self, x, y):
        if x.__class__ != np.ndarray or y.__class__ != np.ndarray or self.thetas.__class__ != np.ndarray:
            return None

        if x.size == 0 or self.thetas.size == 0:
            return None

        if self.thetas.shape[0] != x.shape[1] + 1 or x.shape[0] != y.shape[0]:
            return None

        for _ in range(self.max_iter):
            gradient = self.gradient_(x, y)
            self.thetas -= self.alpha * gradient
        return self.thetas