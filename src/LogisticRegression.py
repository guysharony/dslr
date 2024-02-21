import numpy as np
import matplotlib.pyplot as plt
import time 
from typing import Tuple

class LogisticRegression:
    def __init__(self, learning_rate=0.3, max_iterations=1500, weights=[], bias=[], batch_size=None, multi_class='ovr'):
        """
        Initialize the multinomial logistic regression model.

        Args:
            learning_rate (float): learning rate for gradient descent optimization. Default is 0.1.
            max_iterations (int): maximum number of iterations for training. Default is 1500.
            weights (list): initial weights for the model. Default is an empty list.
            bias (int): initial bias for the model. Default is 0.
            batch_size (int): size of the batch for gradient descent. Defaults to None.
        """
        assert learning_rate is None or (type(learning_rate) in [float, int]) or learning_rate > 0, 'Learning rate must be positive or None'
        assert max_iterations is None or (type(max_iterations) == int) and max_iterations > 0, "Number of maximum iterations must be a positive integer or None"
        assert all(isinstance(weight, float) for weight in weights), 'Weights must be a list of floats'
        assert isinstance(bias, (float, list)) and (isinstance(bias, float) or all(isinstance(b, float) for b in bias)), "Bias must be a float or a list of floats"
        assert multi_class == 'ovr' or multi_class == 'multinomial', "Invalid classification method. Choose between <ovr> or <multinomial>"
        assert batch_size is None or (type(batch_size) == int) or batch_size > 0, "Batch size must be positive or None."
        
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = weights
        self.bias = bias
        self.multi_class = multi_class
        self.batch_size = batch_size


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
    
    def create_batches(self, x, y, i):
        if self.batch_size is None: # Batch gradient descent
            x_batch = x
            y_batch = y
        elif self.batch_size == 1: # Stochastic gradient descent
            random_index = np.random.randint(0, x.shape[0])
            x_batch = x[random_index].reshape(1, -1)
            y_batch = y[random_index].reshape(1, -1)
        else: # Mini-batch gradient descent
            np.random.seed(i)

            indices = np.arange(x.shape[0])
            np.random.shuffle(indices)

            x_shuffled = x[indices]
            y_shuffled = y[indices]

            x_batch = x_shuffled[:self.batch_size]
            y_batch = y_shuffled[:self.batch_size]

        return x_batch, y_batch

    def activation(self, z):
        if self.multi_class == 'ovr':
            return self.sigmoid(z)
        elif self.multi_class == 'multinomial':
            return self.softmax(z)
        raise ValueError('invalid method')

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

    def cross_entropy_loss(self, y_true, y_predictions) -> np.array:
        """
        Compute the binary or multiclass cross-entropy loss 

        Args:
            y_true (numpy.ndarray): Array of true labels in one-hot encoded format.
            y_predictions (numpy.ndarray): Array of predicted probabilities for each class.

        Returns:
            float: cross-entropy loss.
        """
        m = y_true.shape[0]
        epsilon = 1e-15
        y_predictions += epsilon

        if self.multi_class == 'ovr':
            return -(1/m) * np.sum((y_true * np.log(y_predictions)) + (1 - y_true) * np.log(1 - y_predictions))
        return -(1/m) * np.sum(y_true * np.log(y_predictions))
    
    def plot_loss(self, costs):
        print(f'Cost : [{costs[0]}] -> [{costs[-1]}]')

        plt.plot(range(1, self.max_iterations + 1), costs)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')

    def fit(self, x:np.ndarray, y:np.ndarray):
        """
        Train the logistic regression model using the input features (x) and corresponding labels (y).
        Gradient descent optimization to minimize the cross-entropy loss function.
        Sigmoid activation function for one-vs-all classification
        Softmax activation function for multinomial classfication.
        Three types of gradient descent: Batch, Stochastic, and Mini-batch, based on the batch_size parameter.

        Args:
            x (np.ndarray): input features of shape (num_samples, num_features)
            y (np.ndarray): labels corresponding to x encoded as integers

        Returns:
            np.ndarray: Tuple containing the trained weights and bias
        """
        if self.multi_class == 'ovr':
            return self.fit_ovr(x, y)
        elif self.multi_class == 'multinomial':
            return self.fit_multinomial(x, y)
        raise ValueError("Invalid method.")
    
    def fit_ovr(self, x:np.ndarray, y:np.ndarray):
        """
        Fit the logistic regression model using the one-vs-rest method.

        For each class label in the dataset, this method trains a separate logistic regression
        model to distinguish that class from all other classes.

        Args:
            x (np.ndarray): Input features of shape (num_samples, num_features).
            y (np.ndarray): Labels corresponding to x.

        Returns:
            Tuple containing lists of weights and biases for each class.
        """
        classes = np.unique(y)
        weights = np.zeros(x.shape[1])
        bias = 0

        for class_label in classes:
            y_binary = (y == class_label).astype(int)
            weights, bias = self.gradient_descent(x, y_binary)
            self.weights.append(weights)
            self.bias.append(bias)

            self.class_label = class_label

        return self.weights, self.bias

    def gradient_descent(self, x, y):
        weights = np.zeros(x.shape[1])
        bias = 0

        costs = []
        for i in range(self.max_iterations):
            x_batch, y_batch = self.create_batches(x, y, i)

            # Prediction
            y_predictions = self.activation(np.dot(x_batch, weights) + bias)

            # Computing derivative
            dw = (1 / len(x_batch)) * np.dot(x_batch.T, (y_predictions - y_batch.reshape(-1)))
            db = (1 / len(x_batch)) * np.sum(y_predictions - y_batch)

            # Updating weights and bias
            weights -= self.learning_rate * dw
            bias -= self.learning_rate * db

            # Saving cost
            cost = self.cross_entropy_loss(y_batch, y_predictions)
            costs.append(cost)

        self.plot_loss(costs)
        
        return weights, bias

    
    def fit_multinomial(self, x:np.ndarray, y:np.ndarray):
        y_encoded = self.one_hot_encode(y)
        print(y_encoded.shape)
        self.weights = np.zeros((x.shape[1], y_encoded.shape[1]))

        self.bias = 0

        costs = []
        for i in range(self.max_iterations):
            x_batch, y_batch = self.create_batches(x, y_encoded, i)

            # Prediction
            y_predictions = self.activation(x_batch)

            # Computing derivative
            dw = (1 / len(x_batch)) * np.dot(x_batch.T, (y_predictions - y_batch))
            db = (1 / len(x_batch)) * np.sum(y_predictions - y_batch)

            # Updating weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Saving cost
            cost = self.cross_entropy_loss(y_batch, y_predictions)
            costs.append(cost)

        self.plot_loss(costs)

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
        y_predictions = self.softmax(x)
        return np.argmax(y_predictions, axis=1)
