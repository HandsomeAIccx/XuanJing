# -*- coding: utf-8 -*-
# @Time    : 2022/7/16 4:29 下午
# @Author  : Zhiqiang He
# @Email   : tinyzqh@163.com
# @File    : dqn.py
# @Software: PyCharm

import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from XuanJing.buffer.replaybuffer import ReplayBuffer
from XuanJing.utils.torch_utils import tensorify


class DQN(nn.Module):
    def __init__(
            self,
            actor,
            optim,
            args
    ):
        super(DQN, self).__init__()
        self.actor_net = actor.actor_net
        self.target_actor_net = copy.deepcopy(self.actor_net)
        self.optim = optim
        self.args = args
        self.replay_buffer = ReplayBuffer(capacity=args.buffer_size)
        self.learn_step = 0

    def updata_parameter(
            self,
            train_data
    ):
        self.replay_buffer.push(train_data)

        if len(self.replay_buffer) < self.args.start_learn_buffer_size:
            return
        batch_data = self.replay_buffer.random_pop(self.args.batch_size)

        obs = tensorify(batch_data.get_value("obs"))
        actions = tensorify(batch_data.get_value("output")["act"]).view(-1, 1)
        next_obs = tensorify(batch_data.get_value("next_obs"))
        reward = tensorify(batch_data.get_value("reward")).view(-1, 1)
        done = tensorify(batch_data.get_value("done")).view(-1, 1)

        q_values = self.actor_net(obs).gather(1, actions)
        max_next_q_values = self.target_actor_net(next_obs).max(1)[0].view(-1, 1)
        q_targets = reward + self.args.gamma * max_next_q_values * (1 - done)

        loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        if self.learn_step % self.args.update_target_interval == 0:
            self.sync_weight()

        self.learn_step += 1

    def sync_weight(self):
        self.target_actor_net.load_state_dict(self.actor_net.state_dict())

