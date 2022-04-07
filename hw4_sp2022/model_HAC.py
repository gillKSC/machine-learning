""" 
Keep model implementations in here.

This file is where you will write all of your code!
"""

import numpy as np


class Distance_computation_grid(object):
    '''
        class to enable the Computation of distance matrix
    '''

    def __init__(self):
        pass

    def compute_distance(self, samples):
        '''
            Creates a matrix of distances between individual samples and clusters attained at a particular step
        '''
        Distance_mat = np.zeros((len(samples), len(samples)))
        for i in range(Distance_mat.shape[0]):
            for j in range(Distance_mat.shape[0]):
                if i != j:
                    dist = []
                    for m in range(len(samples[i])):
                       for n in range(len(samples[j])):
                           #print(len(samples[i]))
                           #print(samples[i][m])
                           #print(samples[j][n])
                           dist.append(np.linalg.norm(np.array(samples[i][m]) - np.array(samples[j][n])))

                    result = min(dist)
                    Distance_mat[i,j] = float(result)
                    #Distance_mat[i, j] = float(self.distance_calculate(samples[i], samples[j]))
                else:
                    Distance_mat[i, j] = 10 ** 4
        return Distance_mat


class Distance_computation_grid_max(object):
    '''
        class to enable the Computation of distance matrix
    '''

    def __init__(self):
        pass

    def compute_distance(self, samples):
        '''
            Creates a matrix of distances between individual samples and clusters attained at a particular step
        '''
        Distance_mat = np.zeros((len(samples), len(samples)))
        for i in range(Distance_mat.shape[0]):
            for j in range(Distance_mat.shape[0]):
                if i != j:
                    dist = []
                    for m in range(len(samples[i])):
                        for n in range(len(samples[j])):
                            # print(len(samples[i]))
                            # print(samples[i][m])
                            # print(samples[j][n])
                            dist.append(np.linalg.norm(np.array(samples[i][m]) - np.array(samples[j][n])))

                    result = max(dist)
                    Distance_mat[i, j] = float(result)
                    # Distance_mat[i, j] = float(self.distance_calculate(samples[i], samples[j]))
                else:
                    Distance_mat[i, j] = 10 ** 4
        return Distance_mat

class Distance_computation_grid_avg(object):
    '''
        class to enable the Computation of distance matrix
    '''

    def __init__(self):
        pass

    def compute_distance(self, samples):
        '''
            Creates a matrix of distances between individual samples and clusters attained at a particular step
        '''
        Distance_mat = np.zeros((len(samples), len(samples)))
        for i in range(Distance_mat.shape[0]):
            for j in range(Distance_mat.shape[0]):

                if i != j:
                    sum_of_samples = 0.0
                    num = 0.0

                    for k in range(np.shape(samples[i])[0]):
                        for m in range(np.shape(samples[j])[0]):
                            sum_of_samples += float(np.linalg.norm(samples[i][k] - samples[j][m]))
                            num += 1.0
                    #if np.shape(samples[i])[0] == 1 and np.shape(samples[j])[0] ==1 :
                    #    print(sum_of_samples-float(np.linalg.norm(samples[i][0] - samples[j][0])))
                    Distance_mat[i, j] = sum_of_samples/num
                    #print(np.shape(samples[i])[0]*np.shape(samples[j])[0])

                    #Distance_mat[i, j] = float(self.distance_calculate(samples[i], samples[j]))
                else:
                    Distance_mat[i, j] = 10 **6
        return Distance_mat




class Model(object):
    """ Abstract model object."""

    def __init__(self):
        raise NotImplementedError()

    def fit_predict(self, X):
        """ Predict.

        Args:
            X: A dense matrix of floats with shape
                [num_examples, num_features].

        Returns:
            A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()



class AgglomerativeClustering(Model):



    def __init__(self, n_clusters = 2, linkage = 'single'):
        """
        Args:
            n_clusters: number of clusters
            linkage: linkage criterion
        """

        self.stop = n_clusters
        self.link = linkage


    def fit_predict(self, X):
        """ Fit and predict.

        Args:
            X: A dense matrix of floats with shape
                [num_examples, num_features].

        Returns:
            A dense array of ints with shape [num_examples].
        """


        # TODO: Implement this!
        progression = [[i] for i in range(X.shape[0])]

        samples = [[X[i]] for i in range(X.shape[0])]
        m = len(samples)
        distcal = Distance_computation_grid()
        distcal_max = Distance_computation_grid_max()
        distcal_avg = Distance_computation_grid_avg()
        while m > self.stop:

            if self.link == 'single':
                Distance_mat = distcal.compute_distance(samples)
                sample_ind_needed = np.where(Distance_mat == Distance_mat.min())[0]
            if self.link == 'complete':
                Distance_mat = distcal_max.compute_distance(samples)
                sample_ind_needed = np.where(Distance_mat == Distance_mat.min())[0]
            if self.link == 'average':
                Distance_mat = distcal_avg.compute_distance(samples)
                sample_ind_needed = np.where(Distance_mat == Distance_mat.min())[0]
            if np.ndim(sample_ind_needed) == 1:
                min = 10000
                min_x = 10000;
                min_y = 10000;
                problematic = []
                for i in range(Distance_mat.shape[0]):
                    for j in range(Distance_mat.shape[1]):
                        if Distance_mat[i][j] < min:
                            min = Distance_mat[i][j]
                            min_x = i
                            min_y = j
                problematic.append([min_x, min_y])
                for i in range(Distance_mat.shape[0]):
                    for j in range(Distance_mat.shape[1]):
                        if Distance_mat[i][j] == min and (i != min_x or j != min_y):
                            problematic.append([i , j])
                sorted(problematic, key=lambda element: (element[0], element[1]))
                sample_ind_needed = problematic[0]




            #print(np.where(Distance_mat == Distance_mat.min()))

            clus_1_ind = sample_ind_needed[0]

            clus_2_ind = sample_ind_needed[1]
            clus_2 = progression[clus_2_ind]
            samples_clus_2 = samples[clus_2_ind]
            #print(clus_2_ind)
            #if samples_clus_2[0][0] == 31:
            #    print("alert")
            progression[clus_1_ind] = progression[clus_1_ind] + clus_2
            # print(clus_2)
            # print(progression[clus_1_ind])

            samples[clus_1_ind] = samples[clus_1_ind] + samples_clus_2

            progression.pop(clus_2_ind)
            samples.pop(clus_2_ind)

            #print(progression[clus_1_ind])


            m = len(samples)
        ans = np.zeros(X.shape[0])
        for x in range(len(progression)):
            curr_clus = progression[x]
            for y in curr_clus:
                ans[y] = x
        return ans




