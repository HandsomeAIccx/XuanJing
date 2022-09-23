# -*- coding: utf-8 -*-
# @Time    : 4/10/22 11:04 PM
# @Author  : Zhiqiang He
# @Email   : tinyzqh@163.com
# @File    : common.py.py
# @Software: PyCharm

import torch
import numpy as np
from torch import nn

def mlpblock(
		input_dim,
		output_dim,
		activation=None,
):
	layers = [nn.Linear(input_dim, output_dim)]
	if activation is not None:
		layers += [activation()]
	return layers


class MLP(nn.Module):
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

