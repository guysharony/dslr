import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt

from typing import Tuple


class LogisticRegression:
    def __init__(self, learning_rate=0.5, max_iterations=1500, thetas=[], batch_size=None):
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

    def sigmoid(self, z):
        """ Maps data points to the range [0, 1] representing the probability
        of belonging to a certain class in binary classification tasks.

        Args:
            z (np.ndarray): linear combination of input features.
                        weighted sum of the inputs adjusted by the bias 

        Returns:
            np.ndarray: results of sigmoid function 
        """
        return 1 / (1 + np.exp(-z))

    def one_hot_encoding(self, y):
        """ Encodes labels into binary vectors 

        Args:
            y (np.ndarray): labels

        Returns:
            np.ndarray: binary encoded labels
        """
        one_hot_y = np.zeros((len(y), len(np.unique(y))))
        for i in range(len(y)):
            one_hot_y[i, y[i]] = 1
        return one_hot_y

    def create_batch(self, x, y, i):
        """ Create batches for gradient descent to train on

        Args:
            x (np.ndarray): input features
            y (np.ndarray): target labels
            i (int): current iteration index

        Returns:
            tuple: batched x and y 
        """
        m, _ = x.shape

        if self.batch_size is None: # Batch gradient descent
            x_batch = x
            y_batch = y
        elif self.batch_size == 1: # Stochastic gradient descent
            random_index = np.random.randint(0, m)
            x_batch = x[random_index]
            y_batch = y[random_index]
        else: # Mini-batch gradient descent
            np.random.seed(i)

            indices = np.arange(m)
            np.random.shuffle(indices)

            x_shuffled = x[indices]
            y_shuffled = y[indices]

            x_batch = x_shuffled[:self.batch_size]
            y_batch = y_shuffled[:self.batch_size]

        return x_batch, y_batch

    def hypothesis(self, x, thetas):
        """ Compute the activation function

        Args:
            x (np.ndarray): input features
            thetas (np.ndarray): model parameters (weights and bias)

        Returns:
            np.ndarray: predicted probabilities
        """
        return self.sigmoid(np.dot(x, thetas.T))

    def predict(self, x):
        """ Makes predictions using the trained model

        Args:
            x (np.ndarray): input features

        Returns:
            np.ndarray: predicted labels
        """
        class_probabilities = np.zeros((x.shape[0], 4))

        for class_type in range(4):
            x_prime = np.hstack((np.ones((x.shape[0], 1)), x))
            y_hypothesis = self.hypothesis(x_prime, self.thetas[class_type])

            class_probabilities[:, class_type] = y_hypothesis

        return np.argmax(class_probabilities, axis=1)

    def cross_entropy_loss(self, x, y, thetas) -> np.array:
        """ Computes the cross-entropy loss: measures the difference
        between the true binary labels and predicted probabilitites

        Args:
            x (np.ndarray): input features
            y (np.ndarray): target labels
            thetas (np.ndarray): predicted probabilities

        Returns:
            np.array: cross entropy loss
        """
        m, _ = x.shape

        y_hypothesis = self.hypothesis(x, thetas)
        return - (1 / m) * np.sum((y * np.log(y_hypothesis)) + (1 - y) * np.log(1 - y_hypothesis))

    def gradient(self, x, y, y_hypothesis):
        """ Computes the gradient of the loss function

        Args:
            x (np.ndarray): input features
            y (np.ndarray): target labels
            y_hypothesis (np.ndarray): predicted probabilities 

        Returns:
            np.ndarray: gradient of the loss functions
        """
        return x.T.dot(y_hypothesis - y) / len(x)

    def fit(self, x, y):
        """ Fits the one-vs-all logistic regression model to the training data

        Args:
            x (np.ndarray): input features
            y (np.ndarray): target labels

        Returns:
            np.ndarray: model parameters (weights and bias)
        """
        class_types = np.unique(y)

        x_prime = np.hstack((np.ones((x.shape[0], 1)), x))
        y_hot_encoded = self.one_hot_encoding(y)

        self.thetas = rand(class_types.shape[0], x_prime.shape[1])

        for class_type in class_types:

            costs = []
            for i in range(self.max_iterations):
                x_batch, y_batch = self.create_batch(x_prime, y_hot_encoded[:, class_type], i)

                y_hypothesis = self.hypothesis(x_batch, self.thetas[class_type])

                gradient = self.gradient(x_batch, y_batch, y_hypothesis)

                self.thetas[class_type] -= self.learning_rate * gradient

                cost = self.cross_entropy_loss(x_prime, y_hot_encoded[:, class_type], self.thetas[class_type])
                costs.append(cost)

            self.plot_loss(costs)

        return self.thetas

    def plot_loss(self, costs):
        """ Plots the loss function with number of iterations on x-axis and
        loss values on y-axis

        Args:
            costs (list): List of costs over training iterations.
        """
        print(f'Cost : [{costs[0]}] -> [{costs[-1]}]')

        plt.plot(range(1, self.max_iterations + 1), costs)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')

    def accuracy(self, y_prediction, y_true):
        """ Compute the accuracy of the model

        Args:
            y_prediction (numpy.ndarray): predicted labels
            y_true (numpy.ndarray): true labels

        Returns:
            float: accuracy score
        """
        return np.mean(y_prediction == y_true)
