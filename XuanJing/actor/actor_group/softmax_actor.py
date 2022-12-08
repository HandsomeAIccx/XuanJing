import torch
import numpy as np

from XuanJing.actor.actor_group.base import BaseActor
import torch.nn.functional as F


class SoftmaxActor(BaseActor):
    def __init__(self, actor_net, env, args):
        super(SoftmaxActor, self).__init__(actor_net, env, args)
        self.actor_net = actor_net
        self.env = env
        self.args = args

    def sample_forward(self, obs):
        """
        采样的actor调用的前向。
        返回一个字典，字典中包含所有需要记录的网络输出。
        """
        output = {}
        if not isinstance(obs, torch.Tensor):
            obs = torch.Tensor(obs)

        softmax_probs = F.softmax(self.actor_net(obs), dim=1)
        action_dist = torch.distributions.Categorical(softmax_probs)
        action = action_dist.sample()
        output['act'] = np.array([action.item()])
        return output

    def sample_softmax_logits(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.Tensor(obs)
        softmax_probs = F.softmax(self.actor_net(obs), dim=1)
        return softmax_probs
