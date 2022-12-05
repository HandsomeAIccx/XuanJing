import torch
import torch.nn.functional as F
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
		self.fc = MLP(input_dim, output_dim, hidden_sizes)
		self.action_bound = action_bound  # action_bound是环境可以接受的动作最大值

	def forward(self, x):
		x = torch.tanh(self.fc(x))
		return x * self.action_bound


class CarlaNet(torch.nn.Module):
	def __init__(self, env):
		super(CarlaNet, self).__init__()
		self.env = env
		self.bird_eye_shape = env.observation_space.spaces['birdeye'].shape
		self.camera_shape = env.observation_space.spaces['camera'].shape
		self.lidar_shape = env.observation_space.spaces['lidar'].shape
		self.state = env.observation_space.spaces['state'].shape

		# network
		self.state_net = torch.nn.Linear(self.state[0], 2)
		print("a")

	def forward(self, x):
		a = self.state_net(x['state'])
		return a
