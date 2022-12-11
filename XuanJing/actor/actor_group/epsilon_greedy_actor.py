# -*- coding: utf-8 -*-
# @Time    : 2022/9/18 2:44 下午
# @Author  : Zhiqiang He
# @Email   : tinyzqh@163.com
# @File    : epsilon_greedy_actor.py
# @Software: PyCharm

import torch
import numpy as np

from XuanJing.actor.actor_group.base import BaseActor


class EpsGreedyActor(BaseActor):
    def __init__(self, actor_net, env, args):
        super(EpsGreedyActor, self).__init__(actor_net, env, args)
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

        if np.random.random() < self.args.epsilon:
            action = np.array([np.random.randint(np.prod(self.env.action_space.n))])
        else:
            action = self.actor_net(obs).argmax(dim=1).detach().cpu().numpy()

        output['act'] = np.array([action])
        assert len(output['act'].shape) == 2, f"{output['act'].shape} should be 2!"
        return output

