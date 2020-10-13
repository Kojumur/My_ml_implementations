import numpy as np

def scale(X):
    '''Scale numeric data'''
    X = np.array(X)
    for i, column in enumerate(X.T):
        X[:, i] = (column - np.mean(column)) / np.std(column)
    return X