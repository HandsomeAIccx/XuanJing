# -*- coding: utf-8 -*-
# @Time    : 4/10/22 11:04 PM
# @Author  : Zhiqiang He
# @Email   : tinyzqh@163.com
# @File    : common.py.py
# @Software: PyCharm

import torch
import numpy as np
from torch import nn
from torch.nn.utils import vector_to_parameters
from torch.nn.utils import parameters_to_vector


def mlpblock(
		input_dim,
		output_dim,
		activation=None,
):
	layers = [nn.Linear(input_dim, output_dim)]
	if activation is not None:
		layers += [activation()]
	return layers



class Module(nn.Module):
	def __init__(self, **kwargs):
		super(Module, self).__init__()
		for key, val in kwargs.items():
			self.__setattr__(key, val)

	@property
	def num_params(self):
		r"""Return the total number of parameters in the neural network."""
		return sum(param.numel() for param in self.parameters())

	@property
	def num_trainable_params(self):
		r"""Returns the total number of trainable parameters in the neural network."""
		return sum(param.numel() for param in self.parameters() if param.requires_grad)

	@property
	def num_untrainable_params(self):
		r"""Returns the total number of untrainable parameters in the neural network. """
		return sum(param.numel() for param in self.parameters() if not param.requires_grad)

	def to_vec(self):
		r"""Returns the network parameters as a single flattened vector. """
		return parameters_to_vector(parameters=self.parameters())

	def from_vec(self, x):
		r"""Set the network parameters from a single flattened vector.
        Args:
            x (Tensor): A single flattened vector of the network parameters with consistent size.
        """
		vector_to_parameters(vec=x, parameters=self.parameters())


class MLP(Module):
	def __init__(
			self,
			input_dim,
			output_dim,
			hidden_sizes,
			activation=nn.ReLU
	):
		super(MLP, self).__init__()

		activation_list = [activation for _ in range(len(hidden_sizes))]
		hidden_sizes = [input_dim] + list(hidden_sizes)
		model = []
		for in_dim, out_dim, activ in zip(hidden_sizes[:-1], hidden_sizes[1:], activation_list):
			model += mlpblock(input_dim=in_dim, output_dim=out_dim, activation=activ)

		if output_dim > 0:
			model += mlpblock(hidden_sizes[-1], output_dim)

		self.model = nn.Sequential(*model)

	def forward(self, obs):
		assert isinstance(obs, torch.Tensor), "Forward obs data must be Tensor."
		return self.model(obs)

