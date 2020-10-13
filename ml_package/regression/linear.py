from ml_package import regression
from ml_package.regression.base_class import BaseRegression
import numpy as np

class LinearRegression(BaseRegression):
    ''' Linear regression with hypothesis - sigmoid'''

    def compute_hypothesis(self, X):
        '''Computes the hypothesis - linear function'''
        return np.dot(X, self.theta)

    def compute_cost_function(self, X, y, vectorized=True):
        '''Computes the cost function - mean squared error'''
        n = len(y)
        error = np.dot(X, self.theta) - y
        cost = 1 / (2 * n) * np.dot(error.T, error)
        return cost
    
    def normal_eqn(self, X, y):
        '''Analytic solution to gradient descent for linear regression'''
        theta = np.dot(np.dot(np.linalg.pinv(np.dot(X.T,X)), X.T), y)
        return theta

