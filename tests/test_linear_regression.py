from ml_package.regression.linear import LinearRegression
from ml_package.tools.numeric_data_tools import scale
from pathlib import Path
import numpy as np

# Linear regression test
data_path = "tests/data_for_tests/test_linear_3D.txt"
data = np.genfromtxt(Path.cwd().joinpath(data_path), delimiter=',')
regressor = LinearRegression(alfa=0.5)
X = data[:, :-1]
X_scaled = scale(X)
X_scaled = np.reshape(X_scaled, (X_scaled.shape[0], -1))
y = np.array(data[:, -1])
regressor.fit(X_scaled, y)
X_temp = np.hstack((np.ones((len(y), 1) ), X_scaled))
print('\nNormal equation solution =', regressor.normal_eqn(X_temp, y))
print('Gradient descend solution =', regressor.theta)
print('Cost function: ', regressor.compute_cost_function(X_temp,y))

# Plot prediction line from thetas in matplotlib for linear regression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot2D():
    plt.figure()
    plt.scatter(X_scaled, y)
    x_plot = np.array([i for i in np.arange(-1, 5)])
    y_plot = regressor.theta[0] + x_plot * regressor.theta[1] 
    plt.plot(x_plot, y_plot)
    plt.show()

def plot3D():
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], y)
    x_plot = np.array([i for i in np.arange(-3, 3)])
    y_plot = np.array([i for i in np.arange(-3, 3)])
    z_plot = regressor.theta[0] + regressor.theta[1] * x_plot + regressor.theta[2] * y_plot
    ax.plot(x_plot, y_plot, z_plot) 
    plt.show()

# If X has one column plot 2D line 
if X.shape[1] == 1:
    plot2D()

# If X has two columns plot 3D line
elif X.shape[1] == 2:
    plot3D()