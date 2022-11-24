import torch
import numpy as np

from XuanJing.actor.actor_group.base import BaseActor


class EsActor(BaseActor):
    def __init__(self, actor_net, env, args):
        super(EsActor, self).__init__(actor_net, env, args)
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
        action = np.argmax(self.actor_net(obs).detach().numpy())
        # action = 1 if (self.actor_net[0] * obs).sum() > 0 else 0
        output['act'] = np.array([action])
        return output