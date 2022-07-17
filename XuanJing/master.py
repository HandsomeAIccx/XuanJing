# -*- coding: utf-8 -*-
# @Time    : 2022/7/16 1:37 下午
# @Author  : Zhiqiang He
# @Email   : tinyzqh@163.com
# @File    : master.py
# @Software: PyCharm


import gym
import numpy as np
import torch.optim

from utils.net.common import MLP
if __name__ == "__main__":
    from XuanJing.env.vector.vecbase import BaseVectorEnv

    env_num = 6
    envs = BaseVectorEnv([gym.make('CartPole-v1') for _ in range(env_num)])
    envs.reset()
    obs, rew, done, info = envs.step([1] * env_num)

    actor_net = MLP(
        input_dim=int(np.prod(envs.observation_space.shape)),
        output_dim=int(np.prod(envs.action_space.n)),
        hidden_sizes=[128, 128],
    )
    optim = torch.optim.Adam(actor_net.parameters(), lr=1e-3)
    collector =
    # algorithm =

    # enhancefunc =
    # result = learner()
