import numpy as np

def sigmoid(x):
    x = np.clip(x, a_min = -709, a_max = 709)
    return 1 / (1 + np.exp(-x))

def fix_test_feats(X, n_features):
        """ Fixes some feature disparities between datasets.
        Call this before you perform inference to make sure your X features
        match your weights.
        """
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
        print(X.shape)
        X = fix_test_feats(X, self.n_features)
        print(X.shape)
        n, d = X.shape
        
        
        y_hat = np.zeros(n)
        for i in range(n):
            x_p = X[i]
            logits = np.dot(x_p, self.W)
            y_p = sigmoid(logits)

            if y_p >= 0.5:
                y_hat[i] = 1 

        

        y_hat = y_hat.astype(int)

        return y_hat
    
class LogisticRegressionNewton(Model):
    
    def __init__(self, n_features):
        self.n_features = n_features
        self.beta = None

    def fit(self, X, y):
        X = X.todense()
        n_samples, n_features = X.shape

        
        # init parameters
        self.beta = np.zeros(n_features+1)
        one = np.ones((n_samples, 1))
        X = np.column_stack((X, one))
        h = self._sigmoid(np.dot(X, self.beta))
        
        #print(h.shape)
        #print(X.T.shape)
        gradient = np.dot(X.T, (h - y).T) / n_samples
        #print(gradient.shape)
        diag = np.multiply(h.T, (1 - h)) * np.identity(n_samples)
        #print(diag.shape)
        hessian = (1 / n_samples) * np.dot(np.dot(X.T, diag), X)
        #print(hessian.shape)
        subtr = np.dot(np.linalg.pinv(hessian), gradient)
        #print(subtr.shape)
        print(self.beta.shape)
        self.beta = self.beta - subtr
        #print(self.beta.shape)

    def predict(self, X):
        X = X.todense()
        X = fix_test_feats(X, self.n_features+1)
        linear_model = np.dot(X, self.beta)
        y_predicted = self._sigmoid(linear_model)
        
        n, d = X.shape
        y_hat = np.zeros(n)
        
        for i in range(n):
            y_p = y_predicted[i]
            if y_p >= 0.5:
                y_hat[i] = 1 

        

        y_hat = y_hat.astype(int)

        return y_hat

    def _sigmoid(self, x):
        x = np.clip(x, a_min = -709, a_max = 709)
        return 1 / (1 + np.exp(-x))
