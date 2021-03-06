""" 
Keep model implementations in here.

This file is where you will write all of your code!
"""

import numpy as np
from tqdm import tqdm

class Model(object):
    """ Abstract model object.

    Contains a helper function which can help with some of our datasets.
    """

    def __init__(self, nfeatures):
        self.num_input_features = nfeatures


    def fit(self, *, X, iterations):
        """ Fit the model.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            iterations: int giving number of clustering iterations
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


    def _fix_test_feats(self, X):
        """ Fixes some feature disparities between datasets.
        Call this before you perform inference to make sure your X features
        match your weights.
        """
        num_examples, num_input_features = X.shape
        if num_input_features < self.num_input_features:
            X = X.copy()
            X._shape = (num_examples, self.num_input_features)
        if num_input_features > self.num_input_features:
            X = X[:, :self.num_input_features]
        return X


class LambdaMeans(Model):

    def __init__(self, *, nfeatures, lambda0):
        super().__init__(nfeatures)
        """
        Args:
            nfeatures: size of feature space (only needed for _fix_test_feats)
            lambda0: A float giving the default value for lambda
            mu_k: vector of cluster means, the size of this vector will change
        """
        self.lambda0 = lambda0
        self.mu_k = None      


    def fit(self, *, X, iterations):
        """
        Fit the LambdaMeans model.
        Note: labels are not used here.
        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            iterations: int giving number of clustering iterations
        """

        self.mu_k = []
        mean = np.asfarray(X.mean(axis=0))[0]
        self.mu_k.append(mean)

        my_iterations = iterations
        (n, num_features) = X.shape

        if(self.lambda0 == 0):
            #calculate lambda 0 to be the standard deviation of the points
            totalDistanceFromMean = 0
            for x_i in X:
                x_i = x_i.toarray()[0]
                totalDistanceFromMean += self.distance(x_i, mean)
            self.lambda0 = totalDistanceFromMean/n
        
        clusters = []
        clusters.append([]) #start with 1 cluster
        for iteration in range(my_iterations):
            #initialize clusters
            for i in range(len(self.mu_k)):
                if len(clusters[i]) == 0:
                    self.mu_k[i] = mean
                clusters[i] = []
            for i, x_i in enumerate(X):
                x_i = x_i.toarray()[0]
                min_distance = self.distance(x_i, self.mu_k[0])
                min_center_index = 0
                cur_distance = 0
                for center_index, center in enumerate(self.mu_k):
                    cur_distance = self.distance(x_i, center)
                    if cur_distance < min_distance:
                        min_distance = cur_distance
                        min_center_index = center_index
                if(min_distance > self.lambda0):
                    min_distance = 0
                    min_center_index = len(self.mu_k)
                    self.mu_k.append(x_i)
                    #add a new cluster
                    clusters.append([])
                clusters[min_center_index].append(x_i) #put this point in a cluster

            for cluster_index, cluster_points in enumerate(clusters):
                self.mu_k[cluster_index] = np.mean(cluster_points, axis=0)
                
        return self.mu_k, len(clusters)
    
    def distance(self, point1, point2):
        return np.sum(((point1 - point2)**2))**(1/2)

    def predict(self, X):
        """ Predict.
        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
        Returns:
            A dense array of ints with shape [num_examples].
        """

        X = self._fix_test_feats(X)
        (n, num_features) = X.shape
        y = [0] * n
        
        for i, x_i in enumerate(X):
                x_i = x_i.toarray()[0]
                min_distance = self.distance(x_i, self.mu_k[0])
                min_center_index = 0
                cur_distance = 0
                for center_index, center in enumerate(self.mu_k):
                    cur_distance = self.distance(x_i, center)
                    if cur_distance < min_distance:
                        min_distance = cur_distance
                        min_center_index = center_index
                y[i]= min_center_index
        return y
