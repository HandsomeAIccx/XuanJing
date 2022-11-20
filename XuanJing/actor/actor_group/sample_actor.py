import torch
import numpy as np

from XuanJing.actor.actor_group.base import BaseActor
from torch import Tensor
import torch.nn as nn
from torch.distributions.normal import Normal


class SampleActor(BaseActor):
    def __init__(self, actor_net, env, args):
        super(SampleActor, self).__init__(actor_net, env, args)
        self.env = env
        self.args = args
        self.actor_net = actor_net
        self.action_std_log = nn.Parameter(torch.zeros((1, self.env.action_dim)), requires_grad=True)

    def sample_forward(self, obs):
        """
        采样的actor调用的前向。
        返回一个字典，字典中包含所有需要记录的网络输出。
        """
        output = {}
        if not isinstance(obs, torch.Tensor):
            obs = torch.Tensor(obs)

        action_avg = self.actor_net(obs)
        action_std = self.action_std_log.exp()

        dist = Normal(action_avg, action_std)
        action = dist.sample()
        logprob = dist.log_prob(action).sum(1)

        output['act'] = np.array([action.item()])
        output['act_logprob'] = logprob
        return output

    def get_logprob_entropy(self, state: Tensor, action: Tensor) -> (Tensor, Tensor):
        action_avg = self.actor_net(state)
        action_std = self.action_std_log.exp()

        dist = Normal(action_avg, action_std)
        logprob = dist.log_prob(action).sum(1)
        entropy = dist.entropy().sum(1)
        return logprob, entropy