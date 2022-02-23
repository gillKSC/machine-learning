import numpy as np

def sigmoid(x):
    x = np.clip(x, a_min = -709, a_max = 709)
    return 1 / (1 + np.exp(-x))

def fix_test_feats(X, n_features):
    num_examples, num_input_features = X.shape
    if num_input_features < n_features:
        zero = np.zeros((num_examples, 1))
        for i in range(n_features - num_input_features):
            X = np.column_stack((X, zero))
    if num_input_features > n_features:
        X = X[:, :n_features]
    return X
    
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
        num_examples, num_input_features = X.shape

        for i in range(num_examples):
            x_i = X[i, :]
            y_i = y[i]


            logits = np.dot(x_i, self.W)
            h = sigmoid(logits) 

            gradient = np.multiply(x_i, (y_i - h))

            self.W = self.W + self.learning_rate * gradient.T


    def predict(self, X):

        X = X.todense()
        print(X.shape)
        X = fix_test_feats(X, self.n_features)
        print(X.shape)
        num_examples, num_input_features  = X.shape


        y_hat = np.zeros(num_examples)
        for i in range(num_examples):
            x_i = X[i]
            logits = np.dot(x_i, self.W)
            y_i = sigmoid(logits)

            if y_p >= 0.5:
                y_hat[i] = 1 



        y_hat = y_hat.astype(int)

        return y_hat
    
class LogisticRegressionNewton(Model):
    
    def __init__(self, n_features):
        self.n_features = n_features
        self.W = np.zeros((n_features,1))

    def fit(self, X, y):
        X = X.todense()
        n_samples, n_features = X.shape

        h = sigmoid(np.dot(X, self.W))
        y = y.reshape(n_samples,1)
        gradient = np.dot(X.T, (y - h))
        hessian = np.zeros((n_features,n_features))
        for i in range(hessian.shape[0]):
            for j in range(hessian.shape[1]):
                a = np.multiply(h, (1-h))
                b = np.multiply(X[:,i], X[:,j])
                hessia_ij = np.sum(np.multiply(a, b))
                hessian[i,j] = -hessia_ij

        self.W = self.W - np.dot(np.linalg.pinv(hessian), gradient)

    def predict(self, X):

        X = X.todense()
        X = fix_test_feats(X, self.n_features)
        n, d = X.shape

        y_hat = np.zeros(n)
        for i in range(n):
            x_i = X[i]
            logits = np.dot(x_i, self.W)
            y_p = sigmoid(logits)

            if y_p >= 0.5:
                y_hat[i] = 1 

        

        y_hat = y_hat.astype(int)

        return y_hat
