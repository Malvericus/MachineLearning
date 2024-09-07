import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Number of data points
n = 100
X = np.random.randn(100, 1)
y = 5*X + 0.01*np.random.randn(100, 1)
linereg = LinearRegression()
linereg.fit(X, y)
ypred = linereg.predict(X)

plt.plot(X, np.abs(ypred-y)/abs(y), "ro")
plt.axis([0, 1.0, 0.0, 0.5])
plt.xlabel(r'$X$')
plt.ylabel(r'$\epsilon_{\mathrm{relative}}$')
plt.title(r'Relative Error')
plt.show()
