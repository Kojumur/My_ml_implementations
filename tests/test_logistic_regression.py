from ml_package.regression.logistic import LogisticRegression
from ml_package.tools.numeric_data_tools import scale
from pathlib import Path
import numpy as np

# Logistic regression test
data_path = "tests/data_for_tests/test_logistic.txt"
data = np.genfromtxt(Path.cwd().joinpath(data_path), delimiter=',')
regressor = LogisticRegression(alfa=0.5)
X = data[:, :-1]
X_scaled = scale(X)
X_scaled = np.reshape(X_scaled, (X_scaled.shape[0], -1))
y = np.array(data[:, -1])
regressor.fit(X_scaled, y)
X_temp = np.hstack((np.ones((len(y), 1) ), X_scaled))

predictions = [np.round(regressor.compute_hypothesis(X_temp[index])) for index in range(len(y))]
print('grad desc = ', regressor.theta)
print('Times wrong:', np.sum([np.abs(np.round(regressor.compute_hypothesis(X_temp[index]))-y[index]) for index in range(len(y))]))
print('Cost: ', regressor.compute_cost_function(X_temp, y))

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

index_true = [i for i, value in enumerate(y) if value==1]
index_false = [i for i, value in enumerate(y) if value==0]

X_true = X_scaled[index_true]
X_false = X_scaled[index_false]
y_true = y[index_true]
y_false = y[index_false]

print(regressor.theta)
plt.subplot(2,1,1)
plt.scatter(X_true[:, 0], X_true[:, 1], c='green')
plt.scatter(X_false[:, 0], X_false[:, 1], c='red')
x_plot = np.array([i for i in np.arange(-3, 3)])
y_plot = -(regressor.theta[0] + x_plot * regressor.theta[1]) / regressor.theta[2] 
plt.plot(y_plot, x_plot)

index_true_pr = [i for i, value in enumerate(predictions) if value==1]
index_false_pr = [i for i, value in enumerate(predictions) if value==0]

X_true_pr = X_scaled[index_true_pr]
X_false_pr = X_scaled[index_false_pr]
y_true_pr = y[index_true_pr]
y_false_pr = y[index_false_pr]

plt.subplot(2,1,2)
plt.scatter(X_true_pr[:, 0], X_true_pr[:, 1], c='green')
plt.scatter(X_false_pr[:, 0], X_false_pr[:, 1], c='red')
x_plot = np.array([i for i in np.arange(-3, 3)])
y_plot = -(regressor.theta[0] + x_plot * regressor.theta[1]) / regressor.theta[2] 
plt.plot(y_plot, x_plot)
plt.show()
