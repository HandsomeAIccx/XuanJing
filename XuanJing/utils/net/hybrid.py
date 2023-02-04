import torch
import torch.nn as nn
from typing import Type, Dict

from XuanJing.utils.net.common import MLP
from torch.distributions import Normal, Independent


class HybridActor(nn.Module):
    def __init__(
            self,
            input_dim: int,
            action_shape: Dict[str, int],
            hidden_sizes: list,
            activation: Type[torch.nn.Module]
    ) -> None:
        super(HybridActor, self).__init__()

        # Define encoder module, which maps raw state into embedding vector.
        # It could be different for various state, such as Convolution Neural Network (CNN) for image state \
        # and Multilayer perceptron (MLP) for vector state, respectively.
        # $$ y = max(W_2 max(W_1x+b_1, 0) + b_2, 0)$$
        self.encoder = MLP(input_dim, hidden_sizes, activation)
        # Define action_type head module, which outputs discrete logit.
        # $$ y = Wx + b $$
        self.action_type_shape = action_shape['action_type_shape']
        self.action_type_head = nn.Linear(hidden_sizes[-1], self.action_type_shape)
        # Define action_args head module, which outputs corresponding continuous action arguments.
        # $$ \mu = Wx + b $$
        # $$\sigma = e^w$$
        self.action_args_shape = action_shape['action_args_shape']
        self.action_args_mu = nn.Linear(hidden_sizes[-1], self.action_args_shape)
        self.action_args_log_sigma = nn.Parameter(torch.zeros(1, self.action_args_shape))

    def forward(
            self,
            x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # Transform origin state into embedding vector, i.e. $$(B, *) -> (B, N)$$
        embedding_x = self.encoder(x)
        # output discrete action logit.
        logit = self.action_type_head(embedding_x)
        # output the argument mu depending on the embedding vector.
        mu = self.action_args_mu(embedding_x)
        # Utilize broadcast mechanism to make the same shape between log_sigma and mu.
        # ``zeros_like`` operation doesn't pass gradient.
        log_sigma = self.action_args_log_sigma + torch.zeros_like(mu)
        # Utilize exponential operation to produce the actual sigma.
        # $$\sigma = e^w$$
        sigma = torch.exp(log_sigma)

        output = dict()
        output['act_type_logit'] = logit
        output['mu'] = mu
        output['sigma'] = sigma
        return output

    def sample_action(
            self,
            x: torch.Tensor
    ) -> dict:
        """
        The function of testing sampling hybrid action.
        """
        logit = self.forward(x)
        # Transform logit (raw output of discrete policy head, e.g. last fully connected layer) into probability.
        # $$\text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}$$
        prob = torch.softmax(logit['act_type_logit'], dim=-1)
        # Construct categorical distribution. The probability mass function is: $$f(x=i|\boldsymbol{p})=p_i$$
        # <link https://en.wikipedia.org/wiki/Categorical_distribution link>
        discrete_dist = torch.distributions.Categorical(probs=prob)
        # Sample one discrete action type per sample (state input).
        action_type = discrete_dist.sample()

        # Construct gaussian distribution
        # $$X \sim \mathcal{N}(\mu,\,\sigma^{2})$$
        # Its probability density function is: $$f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left( -\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^{\!2}\,\right)$$
        # <link https://en.wikipedia.org/wiki/Normal_distribution link>
        continuous_dist = Normal(logit["mu"], logit["sigma"])
        # Reinterpret ``action_shape`` gaussian distribution into a multivariate gaussian distribution with
        # diagonal convariance matrix.
        # Ensure each event is independent with each other.
        # <link https://pytorch.org/docs/stable/distributions.html#independent link>
        continuous_dist = Independent(continuous_dist, 1)
        # Sample one action args of the shape ``action_shape`` per sample (state input).
        action_args = continuous_dist.sample()

        output = dict()
        output['act_type'] = action_type
        output['act_logit'] = logit['act_type_logit']
        output['act_args'] = action_args
        output['act_args_mu'] = logit['mu']
        output['act_args_sigma'] = logit['sigma']

        return output


def test_sample_action():
    """
    The function of testing sampling hybrid action.
    Construct a hybrid action (parameterized action)
    policy and sample a group of action.
    """
    B, obs_shape, action_shape = 4, 10, {'action_type_shape': 3, 'action_args_shape': 3}
    mask = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
    # Generate state data from uniform distribution in [0, 1].
    state = torch.rand(B, obs_shape)
    # Define hybrid action network with encoder, discrete head and continuous head.
    policy_network = HybridActor(
        input_dim=obs_shape,
        action_shape=action_shape,
        hidden_sizes=[128],
        activation=torch.nn.ReLU
    )
    # Policy network forward procedure, input state and output treetensor-type logit.
    logit = policy_network(state)
    assert isinstance(logit, dict)
    assert logit['act_type_logit'].shape == (B, action_shape['action_type_shape'])
    assert logit['mu'].shape == (B, action_shape['action_args_shape'])
    assert logit['sigma'].shape == (B, action_shape['action_args_shape'])
    # Sample action accoding to corresponding logit part.
    action = policy_network.sample_action(state)
    assert action['act_type'].shape == (B, )
    assert action['act_args_mu'].shape == (B, action_shape['action_args_shape'])
    assert action['act_args_sigma'].shape == (B, action_shape['action_args_shape'])


if __name__ == "__main__":
    test_sample_action()

