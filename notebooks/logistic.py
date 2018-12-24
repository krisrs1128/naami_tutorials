#!/usr/bin/env python

import torch
import numpy as np
import matplotlib.pyplot as plt



def mixture_of_gaussians(n, k=2, pi=None):
    if not pi:
        pi = np.ones(k) / k

    means = np.array([[0, 1], [-3, -3]])
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

n = 500
means, x = mixture_of_gaussians(n)
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


class LogisticRegression(torch.nn.Module):
    def __init__(self, D):
        super(LogisticRegression, self).__init__() # initialize using superclass
        self.linear = torch.nn.Linear(D, 1)
        self.output = torch.nn.Sigmoid()

    def forward(self, x):
        Wx = self.linear(x)
        return self.output(Wx)

model = LogisticRegression(2)
x_tens = torch.as_tensor(x, dtype=torch.float32)
y_tens = torch.as_tensor(y, dtype=torch.float32)
y_tens = y_tens.reshape((n, 1))
y_hat = model.forward(x_tens) # totally random predictions
plt.scatter(x[:, 0], x[:, 1], c=y_hat.detach().numpy().flatten())

torch.nn.BCELoss()(input=torch.Tensor([0.1]), target=torch.Tensor([1.]))

loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# look at a few values
loss_fn(torch.tensor([0.2]), torch.tensor([1.]))
loss_fn(torch.tensor([0.9]), torch.tensor([1.]))
loss_fn(torch.tensor([0.99]), torch.tensor([1.]))

# train the model
for i in range(500):
    y_hat = model(x_tens)
    loss = loss_fn(input=y_hat, target=y_tens)

    optimizer.zero_grad()
    loss.backward()
    print(loss.item())
    optimizer.step()
    if i % 5 == 0:
        fig = plt.figure()
        plt.scatter(x[:, 0], x[:, 1], c=y_hat.detach().numpy().flatten())
        name = "image_{}".format(str(i).rjust(4, "0"))
        fig.savefig(name)


y_hat = model.forward(x_tens) # totally random predictions
plt.scatter(x[:, 0], x[:, 1], c=y_hat.detach().numpy().flatten())
