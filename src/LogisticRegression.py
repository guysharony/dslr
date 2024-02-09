import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

class LogisticRegression:
    def __init__(self, learning_rate=0.1, num_iterations=15000, weights=None, bias=None):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = weights
        self.bias = bias

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

    def softmax(self, X) -> np.array:
        """
        Compute the probabilities of each class,
        ensuring that the sum of probabilities across all classes equals 1.

        Args:
            X (np.array): features input

        Returns:
            np.array: probabilities of each class
        """
        z = np.dot(X, self.weights) + self.bias
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    
    def cross_entropy_loss(self, y_true, y_pred) -> np.array:
        """
        Compute the cross-entropy loss for multinomial logistic regression

        Args:
            y_true (numpy.ndarray): Array of true labels in one-hot encoded format.
            y_pred (numpy.ndarray): Array of predicted probabilities for each class.

        Returns:
            float: cross-entropy loss.
        """
        num_samples = y_true.shape[0]
        # to avoid div by zero
        epsilon = 1e-15
        # to avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.sum(y_true * np.log(y_pred)) / num_samples
        return loss

    def fit(self, X:np.ndarray, y:np.ndarray) -> np.ndarray:
        """
        Train the multinomial logistic regression model

        Args:
            X (np.ndarray): input features
            y (np.ndarray): labels

        Returns:
            np.ndarray: Tuple containing the trained weights and bias
        """
        m, n = X.shape
        y_encoded = self.one_hot_encode(y)
        
        self.weights = np.zeros((X.shape[1], y_encoded.shape[1]))
        self.bias = 0

        costs = []
        for i in range(self.num_iterations):
            y_pred = self.softmax(X)

            dw = (1 / m) * np.dot(X.T, (y_pred - y_encoded))
            db = (1 / m) * np.sum(y_pred - y_encoded, axis=0)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            cost = self.cross_entropy_loss(y_encoded, y_pred)
            costs.append(cost)
            #print(f'cost[{i}]: {cost}')

        plt.plot(range(1, self.num_iterations + 1), costs)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title('Cost vs Iterations')
        plt.show()
        return self.weights, self.bias

    def predict(self, X):
        """
        Makes predictions using the trained weights and bias

        Args:
            X (np.array): input features

        Returns:
            np.array: predicted labels
        """
        y_pred = self.softmax(X)
        return np.argmax(y_pred, axis=1)

