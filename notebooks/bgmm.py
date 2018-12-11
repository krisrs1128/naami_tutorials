# http://pyro.ai/examples/gmm.html

import matplotlib.pyplot as plt
import numpy as np
import pyro
import pyro.optim
import pyro.distributions as dist
import torch

from matplotlib.patches import Ellipse
from pyro import poutine
from pyro.contrib.autoguide import AutoDelta
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate
from torch.distributions import constraints


def model(data):
    # Global variables.
    weights = pyro.param(
        "weights",
        torch.FloatTensor([0.5]),
        constraint=constraints.unit_interval
    )
    scales = pyro.param(
        "scales",
        torch.stack([torch.eye(2), torch.eye(2)]),
        constraint=constraints.positive
    )

    locs = [
        pyro.sample(
            "locs_{}".format(k),
            dist.MultivariateNormal(torch.zeros(2), 2 * torch.eye(2))
        ) for k in range(K)
    ]

    with pyro.iarange("data", data.size(0), 4) as ind:
        # Local variables.
        assignment = pyro.sample(
            "assignment",
            dist.Bernoulli(torch.ones(len(data)) * weights)
        ).to(torch.int64)
        pyro.sample(
            "obs",
            dist.MultivariateNormal(locs[assignment], scales[assignment]),
            obs=data.index_select(ind)
        )

def full_guide(data):
    locs_v = pyro.param("locs_v", torch.randn((2, 2)))
    scales_v = pyro.param(
        "scales_v",
        torch.stack([2 * torch.eye(2), 2 * torch.eye(2)]),
        constraint=constraints.positive
    )

    q_theta = [pyro.sample(
        "locs_{}".format(k),
        dist.MultivariateNormal(locs_v[k], scales_v[k])
    ) for k in range(K)]

    with pyro.iarange("data", data.size(0), 4):
        # Local variables.
        assignment_probs = pyro.param(
            "assignment_probs",
            torch.ones(len(data)) / K,
            constraint=constraints.unit_interval
        )
        pyro.sample(
            "assignment",
            dist.Bernoulli(assignment_probs),
            infer={"enumerate": "sequential"}
        )


def initialize(data):
    pyro.clear_param_store()
    optim = pyro.optim.Adam({"lr": 0.1, "betas": [0.8, 0.99]})
    elbo = TraceEnum_ELBO(max_iarange_nesting=1)
    return SVI(model, full_guide, optim, loss=elbo)


def get_samples(num_samples=100):
    # underlying parameters
    mu1 = torch.tensor([0., 5.])
    sig1 = torch.tensor([[2., 0.], [0., 3.]])
    mu2 = torch.tensor([5., 0.])
    sig2 = torch.tensor([[4., 0.], [0., 1.]])

    # generate samples
    dist1 = dist.MultivariateNormal(mu1, sig1)
    samples1 = [pyro.sample("samples1", dist1) for _ in range(num_samples)]
    dist2 = dist.MultivariateNormal(mu2, sig2)
    samples2 = [pyro.sample("samples2", dist2) for _ in range(num_samples)]

    return torch.cat((torch.stack(samples1), torch.stack(samples2)))


def plot(data, mus=None, sigmas=None, colors="black", figname="fig.png"):
    # Create figure
    fig = plt.figure()

    # Plot data
    x = data[:, 0]
    y = data[:, 1]
    plt.scatter(x, y, 24, c=colors)

    # Plot cluster centers
    if mus is not None:
        x = [float(m[0]) for m in mus]
        y = [float(m[1]) for m in mus]
        plt.scatter(x, y, 99, c="red")

    # Plot ellipses for each cluster
    if sigmas is not None:
        for sig_ix in range(K):
            ax = fig.gca()
            cov = np.array(sigmas[sig_ix])
            lam, v = np.linalg.eig(cov)
            lam = np.sqrt(lam)
            ell = Ellipse(xy=(x[sig_ix], y[sig_ix]),
                          width=lam[0]*4, height=lam[1]*4,
                          angle=np.rad2deg(np.arccos(v[0, 0])),
                          color="blue")
            ell.set_facecolor("none")
            ax.add_artist(ell)

    # Save figure
    fig.savefig(figname)


if __name__ == "__main__":
    pyro.enable_validation(True)
    pyro.set_rng_seed(42)

    # Create our model with a fixed number of components
    K = 2
    n_iter = 300

    data = get_samples()
    svi = initialize(data)

    true_colors = [0] * 100 + [1] * 100
    plot(data, colors=true_colors, figname="pyro_init.png")

    for i in range(n_iter):
        loss = svi.step(data)

        if i % 1 == 0:
            locs_v = pyro.param("locs_v")
            scales = pyro.param("scales")
            weights = pyro.param("weights")

            print(loss)
            print("locs_v: {}".format(locs_v))
            # print("scales: {}".format(scales))
            # print("weights = {}".format(weights))

            assignment_probs = pyro.param("assignment_probs")
            # print("assignments: {}".format(assignment_probs.data))

        # todo plot data and estimates
            assignments = np.uint8(np.round(assignment_probs.data))
            # plot(data, locs_v.data, scales.data, assignments, figname="pyro_iteration_{}.png".format(str(i).zfill(3)))
