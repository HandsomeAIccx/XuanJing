# -*- coding: utf-8 -*-
# @Time    : 2022/9/24 8:52 上午
# @Author  : Zhiqiang He
# @Email   : tinyzqh@163.com
# @File    : reinforce.py
# @Software: PyCharm

import torch
from XuanJing.utils.torch_utils import tensorify
import torch.nn.functional as F


class Reinforce(object):
    def __init__(
            self,
            actor,
            optim,
            args
    ):
        self.actor_net = actor.actor_net
        self.optim = optim
        self.args = args
        self.learn_step = 0

    def updata_parameter(
            self,
            train_data
    ):
        obs = tensorify(train_data.get_value("obs"))
        actions = tensorify(train_data.get_value("output")["act"]).view(-1, 1)
        next_obs = tensorify(train_data.get_value("next_obs"))
        rewards = tensorify(train_data.get_value("reward")).view(-1, 1)
        done = tensorify(train_data.get_value("done")).view(-1, 1)

        G = 0
        self.optim.zero_grad()
        for i in reversed(range(len(obs))):
            reward = rewards[i]
            action = actions[i].view(-1, 1)
            predict_prob = self.actor_net(obs[i]).unsqueeze(0)
            softmax_probs = F.softmax(predict_prob, dim=1)
            log_prob = torch.log(softmax_probs).gather(1, action)
            G = self.args.gamma * G + reward
            loss = -log_prob * G
            loss.backward()
        self.optim.step()
