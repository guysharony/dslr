import numpy as np
import matplotlib.pyplot as plt
import time 

from typing import Tuple

class LogisticRegression:
    def __init__(self, learning_rate=0.5, max_iterations=1500, weights=[], bias=[], batch_size=None, multi_class='ovr'):
        """
        Initialize the multinomial logistic regression model.

        Args:
            learning_rate (float): learning rate for gradient descent optimization. Default is 0.1.
            max_iterations (int): maximum number of iterations for training. Default is 1500.
            weights (list): initial weights for the model. Default is an empty list.
            bias (int): initial bias for the model. Default is 0.
            batch_size (int): size of the batch for gradient descent. Defaults to None.
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = weights
        self.bias = bias
        self.batch_size = batch_size
        self.multi_class = multi_class

    def one_hot_encode(self, y) -> np.array:
        """
        Encodes each category (Hogwart Houses) as a binary vector
        
        Args:
            y (np.array): Array of labels

        Returns:
            np.array: One-hot encoded labels
        """
        one_hot_y = np.zeros((len(y), len(np.unique(y))))
        for i in range(len(y)):
            one_hot_y[i, y[i]] = 1
        return one_hot_y

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def softmax(self, x) -> np.array:
        """
        Compute the probabilities of each class using softmax function.
        Softmax function normalizes data points into a probability distribution

        Args:
            x (np.array): features input

        Returns:
            np.array: probabilities of each class
        """
        z = np.dot(x, self.weights) + self.bias
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def cross_entropy_loss(self, y_true, y_pred) -> np.array:
        """
        Compute the binary or multiclass cross-entropy loss 

        Args:
            y_true (numpy.ndarray): Array of true labels in one-hot encoded format.
            y_pred (numpy.ndarray): Array of predicted probabilities for each class.

        Returns:
            float: cross-entropy loss.
        """
        m = y_true.shape[0]
        # to avoid div by zero
        epsilon = 1e-15
        # to avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        if self.multi_class == 'ovr':
            return -(1/m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        else:
            return -(1/m) * np.sum(y_true * np.log(y_pred))

    def fit(self, x:np.ndarray, y:np.ndarray):
        """
        Train the multinomial logistic regression model using the input features (x) and corresponding labels (y).
        It implements gradient descent optimization to minimize the cross-entropy loss function.
        Multi-class Softmax activation function to compute class probabilities.
        Three types of gradient descent: Batch, Stochastic, and Mini-batch, based on the batch_size parameter.

        Args:
            x (np.ndarray): input features of shape (num_samples, num_features)
            y (np.ndarray): labels corresponding to x encoded as integers

        Returns:
            np.ndarray: Tuple containing the trained weights and bias
        """
        if self.multi_class == 'ovr':
            return self._fit_ovr(x, y)
        elif self.multi_class == 'multinomial':
            return self._fit_multinomial(x, y)
        else:
            raise ValueError("Invalid method.")
    
    def _fit_ovr(self, x:np.ndarray, y:np.ndarray):
        classes = np.unique(y)
        weights_list = []
        bias_list = []
        m, n = x.shape
        weights = np.zeros(n)
        bias = 0
        print(len(x))
        for class_label in classes:
            y_binary = (y == class_label).astype(int)
            weights, bias = self._gradient_descent(x, y_binary)
            self.weights.append(weights)
            self.bias.append(bias)
        return self.weights, self.bias

    def _gradient_descent(self, x, y):
        m, n = x.shape
        weights = np.zeros(n)
        bias = 0

        costs = []
        for _ in range(self.max_iterations):
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

            z = np.dot(x, weights) + bias
            y_pred = self.sigmoid(z)

            dw = (1 / m) * np.dot(x.T, (y_pred - y))
            db = (1 / m) * np.sum(y_pred - y)

            weights -= self.learning_rate * dw
            bias -= self.learning_rate * db
            # Saving cost
            cost = self.cross_entropy_loss(y, y_pred)
            costs.append(cost)

        print(f'Cost : [{costs[0]}] -> [{costs[-1]}]')

        plt.plot(range(1, self.max_iterations + 1), costs)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        
        return weights, bias

    
    def _fit_multinomial(self, x:np.ndarray, y:np.ndarray):
        m, n = x.shape
        y_encoded = self.one_hot_encode(y)

        self.weights = np.zeros((n, y_encoded.shape[1]))
        self.bias = 0

        costs = []
        for i in range(self.max_iterations):
            if self.batch_size is None: # Batch gradient descent
                x_batch = x
                y_batch = y_encoded
            elif self.batch_size == 1: # Stochastic gradient descent
                random_index = np.random.randint(0, m)

                x_batch = x[random_index].reshape(1, -1)
                y_batch = y_encoded[random_index].reshape(1, -1)
            else: # Mini-batch gradient descent
                np.random.seed(i)
                
                indices = np.arange(m)
                np.random.shuffle(indices)

                x_shuffled = x[indices]
                y_shuffled = y_encoded[indices]

                x_batch = x_shuffled[:self.batch_size]
                y_batch = y_shuffled[:self.batch_size]

            # Prediction
            y_prediction = self.softmax(x_batch)

            # Computing derivative
            dw = (1 / len(x_batch)) * np.dot(x_batch.T, (y_prediction - y_batch))
            db = (1 / len(x_batch)) * np.sum(y_prediction - y_batch, axis=0)

            # Updating weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Saving cost
            cost = self.cross_entropy_loss(y_batch, y_prediction)
            costs.append(cost)

        print(f'Cost : [{costs[0]}] -> [{costs[-1]}]')

        plt.plot(range(1, self.max_iterations + 1), costs)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')

        return self.weights, self.bias

    def predict(self, x):
        """
        Makes predictions using the trained weights and bias
        Softmax function is applied to the linear combination of input features and weight and bias term.
        This produces a vector of probabilities representing the likelihood of each class.
        The class with the highest probability is selected as the predicted class for each input sample.
        Args:
            x (np.array): input features

        Returns:
            np.array: predicted labels
        """
        if self.multi_class == 'ovr':
            return self._predict_ovr(x)
        elif self.multi_class == 'multinomial':
            return self._predict_multinomial(x)
        else:
            raise ValueError("Invalid method.")

    def _predict_ovr(self, x):
        num_classes = len(self.weights)
        class_probabilities = np.zeros((x.shape[0], num_classes))

        for i in range(num_classes):
            scores = np.dot(x, self.weights[i]) + self.bias[i]
            class_probabilities[:, i] = self.sigmoid(scores)

        return np.argmax(class_probabilities, axis=1)

    def _predict_multinomial(self, x):
        if np.array(len(self.weights) == 0 or self.bias == 0).any():
            raise ValueError("Weights or bias not initialized. Model must be trained before making predictions.")
        y_pred = self.softmax(x)
        return np.argmax(y_pred, axis=1)
