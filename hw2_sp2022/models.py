import numpy as np

def sigmoid(x):
    x = np.clip(x, a_min = -709, a_max = 709)
    return 1 / (1 + np.exp(-x))

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
        super().__init__()
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.W = np.zeros((n_features, 1))

    def fit(self, X, y):
        X = X.todense()
        n, d = X.shape



        for i in range(n):
            x_p = X[i, :]
            y_p = y[i]
            
            
            logits = np.dot(x_p, self.W)
            h = sigmoid(logits) 

            gradient = np.multiply(x_p, (y_p - h))

            self.W = self.W + self.learning_rate * gradient.T

    
    def predict(self, X):

        X = X.todense()
        n, d = X.shape

        print(n)
        print(self.W.shape)
        y_hat = np.zeros(n)
        for i in range(n):
            x_p = X[i]
            logits = np.dot(x_p, self.W)
            y_p = sigmoid(logits)

            if y_p >= 0.5:
                y_hat[i] = 1 

        

        y_hat = y_hat.astype(int)

        return y_hat
