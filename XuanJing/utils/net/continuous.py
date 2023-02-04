import torch
import numpy as np
import torch.nn as nn
from typing import Type, Dict

from XuanJing.utils.net.common import MLP
from torch.distributions import Normal, Independent


class ContinuousActor(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_sizes: list,
            activation: Type[torch.nn.Module]
    ) -> None:
        super(ContinuousActor, self).__init__()
        self.encoder = MLP(input_dim, hidden_sizes, activation)
        # Define mu module, which is a FC and output the argument mu for gaussian distribution.
        # $$\mu = Wx + b $$
        self.mu = nn.Linear(hidden_sizes[-1], output_dim)
        # Define log_sigma module, which is a learnable parameter but independent to state
        # Here we set is as log_sigma for the convenience of optimization and usage.
        # $$\sigma = e^w$$
        self.log_sigma = nn.Parameter(torch.zeros(1, output_dim))

    def forward(
            self,
            x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # Transform origin state into embedding vector, i.e. $$(B, *) -> (B, N)$$
        embedding_x = self.encoder(x)
        # Output the argument mu depending on the embedding vector, i.e. $$(B, N) -> (B, A)$$
        mu = self.mu(embedding_x)
        # Utilize broadcast mechanism to make the same shape between log_sigma and mu.
        # 'zeros_like' operation doesn't pass gradient.
        log_sigma = self.log_sigma + torch.zeros_like(mu)
        sigma = torch.exp(log_sigma)
        return {'mu': mu, "sigma": sigma}

    def sample_action(self, x: torch.Tensor) -> dict:
        """
        The function of testing sampling continuous action.
        Construct a standard continuous action.
        """
        logit = self.forward(x)
        # Construct gaussian distribution, i.e.
        # $$X \sim \mathcal{N}(\mu,\,\sigma^{2})$$
        # Its probability density function is: $$f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left( -\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^{\!2}\,\right)$$
        # <link https://en.wikipedia.org/wiki/Normal_distribution link>
        dist = Normal(logit['mu'], logit['sigma'])
        # Reinterpret ``action_shape`` gaussian distribution into a multivariate gaussian distribution with diagonal convariance matrix.
        # Ensure each event is independent with each other.
        # <link https://pytorch.org/docs/stable/distributions.html#independent link>
        dist = Independent(dist, 1)
        # Sample one action of the shape ``action_shape`` per sample (state input) and return it.
        sample_action = dist.sample().detach().numpy()

        output = dict()
        output['act'] = sample_action
        output['logit'] = logit
        return output


def test_sample_action():
    # Set batch_size = 4, obs_shape = 10, action_shape = 6.
    # ``action_shape`` is different from discrete and continuous action. The former is the possible
    # choice of a discrete action while the latter is the dimension of continuous action.
    B, obs_shape, action_shape = 4, 10, 6
    # Generate state data from uniform distribution in [0, 1].
    state = torch.rand(B, obs_shape)
    # Define continuous action network (which is similar to reparameterization) with encoder, mu and log_sigma.
    policy_network = ContinuousActor(
        input_dim=obs_shape,
        output_dim=action_shape,
        hidden_sizes=[128],
        activation=torch.nn.ReLU
    )
    # Policy network forward procedure, input state and output dict-type logit.
    # $$ \mu, \sigma = \pi(a|s)$$
    logit = policy_network(state)
    assert isinstance(logit, dict)
    assert logit['mu'].shape == (B, action_shape)
    assert logit['sigma'].shape == (B, action_shape)
    # Sample action according to corresponding logit (i.e., mu, and sigma).
    action = policy_network.sample_action(state)
    assert action['act'].shape == (B, action_shape)


if __name__ == "__main__":
    test_sample_action()

