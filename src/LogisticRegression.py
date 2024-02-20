import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple


class LogisticRegression:
    def __init__(self, learning_rate=0.1, max_iterations=10, thetas=[], batch_size=None, multi_class='ovr'):
        """
        Initialize the multinomial logistic regression model.

        Args:
            learning_rate (float): learning rate for gradient descent optimization. Default is 0.1.
            max_iterations (int): maximum number of iterations for training. Default is 1500.
            thetas (list): initial weights for the model. Default is an empty list.
            bias (int): initial bias for the model. Default is 0.
            batch_size (int): size of the batch for gradient descent. Defaults to None.
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.thetas = thetas
        self.batch_size = batch_size
        self.multi_class = multi_class

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def gradient_descent(self, x, y):
        m, n = x.shape

        for i in range(self.max_iterations):
            if self.batch_size is None: # Batch gradient descent
                x_batch = x
                y_batch = y
            elif self.batch_size == 1: # Stochastic gradient descent
                random_index = np.random.randint(0, m)
                x_batch = x[random_index].reshape(1, -1)
                y_batch = y[random_index].reshape(1, -1)
            else: # Mini-batch gradient descent
                np.random.seed(i)

                indices = np.arange(m)
                np.random.shuffle(indices)

                x_shuffled = x[indices]
                y_shuffled = y[indices]

                x_batch = x_shuffled[:self.batch_size]
                y_batch = y_shuffled[:self.batch_size]

            

    def fit(self, x, y):
        class_types = np.unique(y)
        m, n = x.shape

        x_prime = np.hstack((np.ones((m, 1)), x))
        self.thetas = np.zeros((x_prime.shape[1], 1))

        print(self.thetas.shape)

        for class_type in class_types:
            y_binary = (y == class_type).astype(int)

            for i in range(self.max_iterations):
                if self.batch_size is None: # Batch gradient descent
                    x_batch = x_prime
                    y_batch = y_binary
                elif self.batch_size == 1: # Stochastic gradient descent
                    random_index = np.random.randint(0, m)
                    x_batch = x_prime[random_index].reshape(1, -1)
                    y_batch = y_binary[random_index].reshape(1, -1)
                else: # Mini-batch gradient descent
                    np.random.seed(i)

                    indices = np.arange(m)
                    np.random.shuffle(indices)

                    x_shuffled = x_prime[indices]
                    y_shuffled = y_binary[indices]

                    x_batch = x_shuffled[:self.batch_size]
                    y_batch = y_shuffled[:self.batch_size]

                y_prediction = self.sigmoid(np.dot(x_batch, self.thetas))

                print(y_prediction)