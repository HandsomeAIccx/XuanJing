import torch
import numpy as np
import torch.nn as nn
from typing import Type

from XuanJing.utils.net.common import MLP


class DiscreteActor(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_sizes: list,
            activation: Type[torch.nn.Module]
    ) -> None:
        super(DiscreteActor, self).__init__()
        self.encoder = MLP(input_dim, hidden_sizes, activation)
        self.head = nn.Linear(hidden_sizes[-1], output_dim)

    def forward(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:
        # Transform origin state into embedding vector, i.e. $$(B, *) -> (B, N)$$
        embedding_x = self.encoder(x)
        # Calculate logit for each possible discrete action choice, i,e, $$(B, N) -> (B, A)$$
        logit = self.head(embedding_x)
        return logit

    def sample_action(self, x: torch.Tensor) -> dict:
        """
        The function of sampling discrete action, input shape = (B, action_shape),
        output shape = (B, ).
        """
        logit = self.forward(x)
        # Transform logit (raw output of policy network, e.g. last fully connected layer) into probability.
        # $$\text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}$$
        prob = torch.softmax(logit, dim=-1)
        # Construct categorical distribution. The probability mass function is: $$f(x=i|\boldsymbol{p})=p_i$$
        # <link https://en.wikipedia.org/wiki/Categorical_distribution link>
        dist = torch.distributions.Categorical(probs=prob)
        # Sample one discrete action per sample (state input) and return it.
        sample_action = dist.sample().item()

        output = {}
        output['act'] = np.array([sample_action])[..., np.newaxis]
        output['logit'] = logit.detach().numpy()
        return output


def test_sample_action():
    """
    The function of testing sampling discrete action.
    Construct a naive policy and sample a group of action.
    """
    # Set batch_size = 4, obs_shape = 10, action_shape = 6.
    B, obs_shape, action_shape = 4, 10, 6
    # Generate state data from uniform distribution in [0, 1].
    state = torch.rand(B, obs_shape)
    # Define policy network with encoder and head.
    policy_network = DiscreteActor(
        input_dim=obs_shape,
        output_dim=action_shape,
        hidden_sizes=[128],
        activation=torch.nn.ReLU
    )
    # Policy network forward procedure, input state and output logit.
    # $$ logit = \pi(a|s)$$
    logit = policy_network(state)
    assert logit.shape == (B, action_shape)
    # Sample action according to corresponding logit.
    action = policy_network.sample_action(state)
    assert action.shape == (B, )


if __name__ == "__main__":
    test_sample_action()

