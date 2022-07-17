# -*- coding: utf-8 -*-
# @Time    : 2022/7/16 4:29 下午
# @Author  : Zhiqiang He
# @Email   : tinyzqh@163.com
# @File    : dqn.py
# @Software: PyCharm

import torch.nn as nn


class DQN(nn.Module):
    def __init__(
            self,
            actor_model,
            optim,
            args,
    ):
        super(DQN, self).__init__()
        self.actor_model = actor_model
        self.optim = optim
        self.args = args

    def compute_loss(
            self,

    ):
        pass


