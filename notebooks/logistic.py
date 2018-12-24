#!/usr/bin/env python

import torch
import numpy as np
import matplotlib.pyplot as plt

def mixture_of_gaussians(n, k=2, pi=None):
    if not pi:
        pi = np.ones(k) / k

    means = np.random.normal(0, 3, (k, 2))
    z = np.random.choice(range(k), n, True, pi)

    data = np.zeros((n, 2))
    for i in range(n):
        data[i, ] = np.random.normal(means[z[i]])

    return means, data

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# mixture of gaussians data
def mog_classes(x, theta, k=2):
    n, _ = x.shape
    x_tilde = np.hstack((np.ones((n, 1)), x))
    pi = sigmoid(np.dot(x_tilde, theta))

    y = np.zeros(n)
    for i in range(n):
        y[i] = np.random.choice(range(k), size=1, p=[1 - pi[i], pi[i]])

    return pi, y


means, x = mixture_of_gaussians(1000)
plt.scatter(x[:, 0], x[:, 1])

theta = means[1, :] - means[0, :]
theta = np.hstack((0.2, theta))
theta /= np.sqrt(sum(theta ** 2))

pi, y = mog_classes(x, theta)
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.scatter(x[:, 0], x[:, 1], c=pi)

# fit a logistic regression on this

# now do a nonlinear transformation
z = np.vstack([x[:, 0] ** 2 - x[:, 1] * x[:, 0], x[:, 0] ** 2 + x[:, 1] ** 2 - 2 * x[:, 0] * x[:, 1]]).T
plt.scatter(z[:, 0], z[:, 1], c=pi)

# fit a logistic regression on this new z
# see if you can add a hidden layer to the logistic model to get a better
# decision boundary
