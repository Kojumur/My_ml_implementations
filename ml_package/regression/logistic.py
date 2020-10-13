from ml_package import regression
from ml_package.regression.base_class import BaseRegression
import numpy as np

class LogisticRegression(BaseRegression):
    '''Logistic regression with cost function - mean squared error'''

    def compute_hypothesis(self, X):
        '''Computes the hypothesis - sigmoid of linear function'''
        return 1 / (1 + np.exp(-np.dot(X, self.theta)))

    def compute_cost_function(self, X, y):
        '''Computes the logarithmic cost function'''
        n = len(y)
        cost = -(1/n) * np.sum(
            y * np.log(self.compute_hypothesis(X)) + (1 - y) * np.log(
                1 - self.compute_hypothesis(X)))
        return cost