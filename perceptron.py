import numpy as np


def activation_f(x):
    return np.where(x >= 0, 1, 0)


class P_model:
    def __init__(self):
        self.Bias = 0
        self.learning_rate = 0.1
        self.steps = 500
        self.activation_f = activation_f
        self.weights = 0

    def prediction(self, X):
        output = np.dot(X, self.weights) + self.Bias
        predicted = self.activation_f(output)
        return predicted

    def fit(self, X, y):
        self.Bias = 1
        sample, features = X.shape
        self.weights = np.zeros(features)

        main_array = np.array([1 if i > 0 else 0 for i in y])

        for i in range(self.steps):
            for index, value in enumerate(X):
                output = np.dot(value, self.weights) + self.Bias
                prediction = self.activation_f(output)

                new_weights = self.learning_rate * (main_array[index] - prediction)
                self.Bias += new_weights
                self.weights += new_weights * value

# End of the perceptron program
