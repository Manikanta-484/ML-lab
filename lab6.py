import numpy as np
import matplotlib.pyplot as plt

def gaussian(x, xi, tau): 
    return np.exp(-np.sum((x - xi)**2) / (2 * tau**2))

def lwr(x, X, y, tau):
    w = np.array([gaussian(x, xi, tau) for xi in X])
    W = np.diag(w)
    theta = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y
    return x @ theta

np.random.seed(42)
X = np.linspace(0, 2*np.pi, 100)
y = np.sin(X) + 0.1*np.random.randn(100)
X_bias = np.c_[np.ones_like(X), X]

x_test = np.c_[np.ones(200), np.linspace(0, 2*np.pi, 200)]
tau = 0.5
y_pred = np.array([lwr(xi, X_bias, y, tau) for xi in x_test])

plt.scatter(X, y, c='r', alpha=0.7, label='Train')
plt.plot(x_test[:,1], y_pred, c='b', lw=2, label=f'LWR tau={tau}')
plt.xlabel('X'); plt.ylabel('y'); plt.title('Locally Weighted Regression')
plt.legend(); plt.grid(alpha=0.3); plt.show()
