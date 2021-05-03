# %%
import torch
from torch import nn

import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
from pyro.infer import Predictive


class HierarchicalBayesianRegression(nn.Module):
    def __init__(self, input_dim):
        super(HierarchicalBayesianRegression, self).__init__()
        self.input_dim = input_dim
        self.mcmc_kernel = None
        self.mcmc = None
        self.posterior_weight_samples = None
        self.num_samples = 1000

    def model(self, x, y):
        # hyper-prior
        alpha = pyro.sample("alpha", dist.Gamma(10e-6, 10e-6))
        sigma = pyro.sample("sigma", dist.Gamma(10e-6, 10e-6))
        # prior
        bias = pyro.sample("bias", dist.Normal(0., 10.))
        w_prior_mean = torch.zeros(self.input_dim, 1)
        w_prior_std = torch.sqrt(1./alpha) * torch.ones(self.input_dim, 1)
        w = pyro.sample("w",
                        dist.Normal(w_prior_mean, w_prior_std).to_event(2)
                        )
        mean = (torch.matmul(x, w) + bias).squeeze()
        with pyro.plate("data", len(x)):
            pyro.sample("obs", dist.Normal(mean, torch.sqrt(1.0/sigma)), obs=y)

    def train_model(self, x, y, num_samples=1000, warmup_steps=200):
        self.num_samples = num_samples
        self.mcmc_kernel = NUTS(self.model)
        self.mcmc = MCMC(self.mcmc_kernel, num_samples=self.num_samples, warmup_steps=warmup_steps)
        self.mcmc.run(x, y)
        self.posterior_weight_samples = self.mcmc.get_samples()

    def make_prediction(self, x):
        mcmc_predictive_model = Predictive(self.model, posterior_samples=self.posterior_weight_samples,
                                           num_samples=self.num_samples,
                                           return_sites=("obs",))
        predictive_samples = mcmc_predictive_model(x, None)['obs'].permute(1, 0)
        return predictive_samples
