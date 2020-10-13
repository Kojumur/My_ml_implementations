import numpy as np
import random

class BaseRegression:
    '''Base class for regression'''

    def __init__(self, alfa=0.001):
        '''Initializing thetas - weights of linear regression, and alfa - learning rate'''
        self.theta = 0
        self.alfa = alfa

    def fit(self, X, y):
        '''Fit weights of linear regression to the data using gradient descent'''
        n = len(y)
        iter = 0

        # Adding first column of ones to X is required to calculate the dot(theta, X)
        x0 = np.ones((n, 1))
        X = np.hstack((x0, X))

        # Initial values for weights
        self.theta = np.array([random.uniform(-10, 10) for i in range(X.shape[1])])
        cost = self.compute_cost_function(X, y)
        print(f'Initial cost function: {cost}\n')
  
        while True:
            iter += 1
            y_pred = self.compute_hypothesis(X)
            error = y_pred - y
            gradient_vector = np.dot(X.T, error)
            self.theta -= self.alfa / n * gradient_vector
            cost_new = self.compute_cost_function(X, y)
            dif_cost = np.sum(np.abs(gradient_vector))
            if iter % 1000 == 0:
                    print(f'iteration: {iter:<8} dif_cost: {dif_cost:.12f}   thetas: {self.theta}')
            cost = cost_new

            # Stop condition - if sum of absolute values of partial derivatives of cost function is close to zero 
            if dif_cost < 10**(-6):
                print(f'Total number of iterations: {iter} \nFinal derivative of cost funcion: {dif_cost} \nFinal cost function: {cost}')
                break