from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import math
import os
import pyro
import pyro.distributions as dist
import torch
import torch.distributions.constraints as constraints
pyro.clear_param_store()

def model(data):
    # define the hyperparameters that control the beta prior
    alpha0 = torch.tensor(10.0)
    beta0 = torch.tensor(10.0)

    # sample f from the beta prior, then draw from the likelihood
    p = pyro.sample("heads_prob", dist.Beta(alpha0, beta0))
    with pyro.iarange("data", len(data)):
        pyro.sample(
            "obs",
            dist.Bernoulli(p).expand_by(data.shape).independent(1),
            obs=data
        )


def guide(data):
    # register the two variational parameters with Pyro.
    alpha_q = pyro.param(
        "alpha_q",
        torch.tensor(15.0),
        constraint=constraints.positive
    )
    beta_q = pyro.param(
        "beta_q",
        torch.tensor(15.0),
        constraint=constraints.positive
    )
    # sample heads_prob from the distribution Beta(alpha_q, beta_q)
    pyro.sample("heads_prob", dist.Beta(alpha_q, beta_q))


# generate data and set up optimizer
data = torch.tensor([1.0] * 20 + [0.0] * 10)
optimizer = Adam({"lr": 0.0005, "betas": (0.90, 0.999)})

# setup the inference algorithm
svi = SVI(model, guide, optimizer, loss=Trace_ELBO(num_particles=10))

n_steps = 5000
# do gradient steps
for step in range(n_steps):
    loss = svi.step(data)
    if step % 100 == 0:
        print(loss)

print(pyro.param("alpha_q").item())
print(pyro.param("beta_q").item())

# true posterior mean
(20 + 10) / (30 + 20)

# estimated posterior mean
pyro.param("beta_q") / pyro.param("alpha_q")
