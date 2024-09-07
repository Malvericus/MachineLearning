import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = np.random.rand(100, 1)
y = 2*X + np.random.rand(100, 1)
linreg = LinearRegression()
linreg.fit(X, y)

# This is our new X-array to which we test our model
X_new = np.array([[0], [1]])
y_pred = linreg.predict(X_new)

plt.plot(X_new, y_pred, "r-")
plt.plot(X, y, 'ro')
plt.axis([0, 1.0, 0, 5.0])
plt.xlabel(r'$X$')
plt.ylabel(r'$y$')
plt.title(r'Simple Linear Regression')
plt.show()