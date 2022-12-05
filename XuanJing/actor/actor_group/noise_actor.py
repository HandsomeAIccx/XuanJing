import torch
import numpy as np
from XuanJing.actor.actor_group.base import BaseActor
from XuanJing.utils.torch_utils import tensorify


class NoiseActor(BaseActor):
    def __init__(self, actor_net, env, args):
        super(NoiseActor, self).__init__(actor_net, env, args)
        self.actor_net = actor_net
        self.env = env
        self.args = args
        self.sigma = args.sigma
        self.action_dim = int(np.prod(env.action_space.shape))
        self.action_bound = env.action_space.high[0]

    def sample_forward(self, obs):
        """
        采样的actor调用的前向。
        返回一个字典，字典中包含所有需要记录的网络输出。
        """
        output = {}
        if not isinstance(obs, torch.Tensor):
            obs = tensorify(obs)
        action = self.actor_net(obs).detach().numpy()
        action = action + self.sigma * np.random.randn(self.action_dim)
        output['act'] = np.array([action])
        return output