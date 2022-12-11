import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
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


def linear_lr_scheduler(optimizer, N, min_lr):
	r"""Defines a linear learning rate scheduler.

	Args:
		optimizer (Optimizer): optimizer
		N (int): maximum bounds for the scheduling iteration
			e.g. total number of epochs, iterations or time steps.
		min_lr (float): lower bound of learning rate
	"""
	initial_lr = optimizer.defaults['lr']
	f = lambda n: max(min_lr / initial_lr, 1 - n / N)
	lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
	return lr_scheduler


def orthogonal_init(module, nonlinearity=None, weight_scale=1.0, constant_bias=0.0):
	r"""Applies orthogonal initialization for the parameters of a given module.

	 Args:
		module (nn.Module): A module to apply orthogonal initialization over its parameters.
		nonlinearity (str, optional): Nonlinearity followed by forward pass of the module. When nonlinearity
			is not ``None``, the gain will be calculated and :attr:`weight_scale` will be ignored.
				Default: ``None``
		weight_scale (float, optional): Scaling factor to initialize the weight. Ignored when
			:attr:`nonlinearity` is not ``None``. Default: 1.0
		constant_bias (float, optional): Constant value to initialize the bias. Default: 0.0

	.. note::

		Currently, the only supported :attr:`module` are elementary neural network layers, e.g.
		nn.Linear, nn.Conv2d, nn.LSTM. The submodules are not supported.

	Example::
		>>> a = nn.Linear(2, 3)
		>>> ortho_init(a)
	"""
	if nonlinearity is not None:
		gain = nn.init.calculate_gain(nonlinearity)
	else:
		gain = weight_scale

	if isinstance(module, (nn.RNNBase, nn.RNNCellBase)):
		for name, param in module.named_parameters():
			if 'weight_' in name:
				nn.init.orthogonal_(param, gain=gain)
			elif 'bias_' in name:
				nn.init.constant_(param, constant_bias)
	else:  # other modules with single .weight and .bias
		nn.init.orthogonal_(module.weight, gain=gain)
		nn.init.constant_(module.bias, constant_bias)


class MLP(Module):
	def __init__(
			self,
			input_dim,
			output_dim,
			hidden_sizes,
			activation=nn.ReLU
	):
		super(MLP, self).__init__()
		assert isinstance(hidden_sizes, list), "hidden_layer_sizes must be a list of integers"
		self.input_dim = input_dim
		self.output_dim = output_dim
		hidden_sizes = [input_dim] + hidden_sizes

		activation_list = [activation for _ in range(len(hidden_sizes))]
		model = []
		for in_dim, out_dim, activ in zip(hidden_sizes[:-1], hidden_sizes[1:], activation_list):
			model += mlpblock(input_dim=in_dim, output_dim=out_dim, activation=activ)

		if output_dim > 0:
			model += mlpblock(hidden_sizes[-1], output_dim)

		self.model = nn.Sequential(*model)

	def forward(self, obs):
		assert isinstance(obs, torch.Tensor), "Forward obs data must be Tensor."
		return self.model(obs)


class VAnet(torch.nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim):
		super(VAnet, self).__init__()
		self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
		self.fc_A = torch.nn.Linear(hidden_dim, output_dim)
		self.fc_v = torch.nn.Linear(hidden_dim, 1)

	def forward(self, x):
		A = self.fc_A(F.relu(self.fc1(x)))
		V = self.fc_v(F.relu(self.fc1(x)))
		Q = V + A - A.mean(1).view(-1, 1)
		return Q


class PolicyNet(torch.nn.Module):
	def __init__(self, input_dim, hidden_sizes, output_dim, action_bound):
		super(PolicyNet, self).__init__()
		self.pre_process = nn.Sequential(
			nn.Linear(input_dim, input_dim)
		)
		self.fc = MLP(input_dim, output_dim, hidden_sizes)
		self.action_bound = action_bound  # action_bound是环境可以接受的动作最大值

	def forward(self, x):
		x = self.pre_process(x)
		x = torch.tanh(self.fc(x))
		return x * self.action_bound


class CarlaNetPreProcess(torch.nn.Module):
	def __init__(self, env):
		super(CarlaNetPreProcess, self).__init__()
		self.bird_eye_shape = env.observation_space.spaces['birdeye'].shape
		self.camera_shape = env.observation_space.spaces['camera'].shape
		self.lidar_shape = env.observation_space.spaces['lidar'].shape
		self.state = env.observation_space.spaces['state'].shape

		self.pre_process = torch.nn.Linear(self.state[0], self.state[0])
		self.output_dim = self.state[0]

	def forward(self, x):
		return self.pre_process(x['state'])


class CarlaNet(torch.nn.Module):
	def __init__(self, env):
		super(CarlaNet, self).__init__()
		self.env = env

		# first step precess the special data struct.
		self.pre_process = CarlaNetPreProcess(env=env)
		# constrain the network you wanted.
		self.state_net = torch.nn.Linear(self.pre_process.output_dim, 2)

	def forward(self, x):
		pre_res = self.pre_process(x)
		return self.state_net(pre_res)
