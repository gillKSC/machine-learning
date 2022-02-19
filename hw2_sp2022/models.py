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

    def __init__(self, n_features, learning_rate = 0.1):
        super().__init__()
        self.n_features = n_features
        self.learning_rate = learning_rate
        pass

    def fit(self, X, y):
        m = np.zeros_like(X[0])
        epochs = 50  # no. of iterations for optimization

        # Performing Gradient Descent Optimization
        # for every epoch
        for epoch in range(1, epochs + 1):
            # for every data point(X_train,y_train)
            for i in range(len(X)):
                # compute gradient w.r.t 'm'
                gr_wrt_m = X[i] * (y[i] - sigmoid(np.dot(self.n_features.T, X[i])))
                # update m, c
                self.n_features = self.n_features - self.learning_rate * gr_wrt_m
        pass

    def predict(self, X):
        predictions = []
        for i in range(len(X)):
            z = np.dot(self.n_features, X[i])
            y_pred = sigmoid(z)
            if y_pred >= 0.5:
                predictions.append(1)
            else:
                predictions.append(0)
        return predictions

class LogisticRegressionNewton(Model):

    def __init__(self, n_features):
        super().__init__()
        # TODO: Initialize parameters
        pass

    def fit(self, X, y):
        # TODO: Write code to fit the model
        pass

    def predict(self, X):
        # TODO: Write code to make predictions
        pass
