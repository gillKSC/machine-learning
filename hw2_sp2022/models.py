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
            x_p = X[i]
            y_p = y[i]

            logits = np.dot(self.W, x_p)
            h = sigmoid(logits)


            gradient = np.dot(x_p.T, (y_p - h))

        self.W += self.learning_rate * gradient

    
    def _fix_test_feats(self, X):
        """ Fixes some feature disparities between datasets.
        Call this before you perform inference to make sure your X features
        match your weights.
        """
        num_examples, num_input_features = X.shape
        if num_input_features < self.n_features:
            X = X.copy()
            X._shape = (num_examples, self.n_features)
        if num_input_features > self.n_features:
            X = X[:, :self.n_features]
        return X
    
    def predict(self, X):

        
        X = X.todense()
        logits = np.dot(self.W, X)
        y_hat = sigmoid(logits)

        for idx in range(len(y_hat)):
            y_hat[idx] = 1 if y_hat[idx] > 0.5 else 0

        y_hat = np.squeeze(np.asarray(y_hat))

        y_hat = y_hat.astype(int)

        return y_hat
