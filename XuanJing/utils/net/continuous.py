import torch
import numpy as np
import torch.nn as nn
from typing import Type

from XuanJing.utils.net.common import MLP


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

        self.head = nn.Linear(hidden_sizes[-1], output_dim)
