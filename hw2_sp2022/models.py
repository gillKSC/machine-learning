import numpy as np



class Model(object):

    def __init__(self):
        self.num_input_features = None

    def fit(self, X, y):
        """ Fit the model.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            y: A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()

    def predict(self, X):
        """ Predict.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].

        Returns:
            A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()

class LogisticRegressionSGD(Model):

    def __init__(self, n_features, learning_rate = 0.01):
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.n_iters = 1000
        self.theta = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.bias = 0
        self.theta = np.zeros((n_features, 1))

        for _ in range(self.n_iters):
            linear_model = self.bias + np.dot(self.theta, X.T)
            y_pred = self._sigmoid(linear_model)

            dw = np.dot((y_pred - y), X) / n_samples
            db = np.sum(y_pred - y) / n_samples

            self.theta -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = self.bias + np.dot(self.theta, X.T)
        y_pred = self._sigmoid(linear_model)[0]
        return [1 if i > 0.5 else 0 for i in y_pred]

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
