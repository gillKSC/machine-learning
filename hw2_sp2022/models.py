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
        
        n, d = X.shape

        X = X.todense()

        for i in range(n):
            x_p = X[i, :]
            y_p = y[i]
            
            
            logits = np.dot(x_p, self.W)
            h = sigmoid(logits) 

            for j in range(self.n_features):
                
                gradient = np.dot(x_p[:,j], (y_p - h))

                self.W[j] = self.W[j] + self.learning_rate * gradient


    
    def predict(self, X):

        
        n, d = X.shape
        X = X.todense()
        
        y_hat = np.zeros(n)
        for i in range(n):
            x_p = X[i, :]
            logits = np.dot(x_p, self.W)
            y_p = sigmoid(logits)

            y_hat[i] = 1 if y_p >= 0.5 else 0

        y_hat = np.squeeze(np.asarray(y_hat))

        y_hat = y_hat.astype(int)

        return y_hat
