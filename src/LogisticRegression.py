import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, learning_rate=0.001, num_iterations=500):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None        

    def sigmoid(self, X):
        z = np.dot(X, self.weights) + self.bias
        return 1 / (1 + np.exp(-z))
    

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0
        
        costs = []
        for i in range(self.num_iterations):
            y_pred = self.sigmoid(X)

            dw = (1 / m) * np.dot(X.T, (y_pred - y))
            db = (1 / m) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            cost = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
            costs.append(cost)
            print(f'Iteration {i+1}, Cost: {cost}')
        # print(f'WEIGHT:{self.weights}')
        # Optionally return the final cost for monitoring convergence
        # Plot costs
        plt.plot(range(1, self.num_iterations + 1), costs)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title('Cost vs Iterations')
        plt.show()

        return costs

