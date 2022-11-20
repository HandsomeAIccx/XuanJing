# -*- coding: utf-8 -*-
# @Time    : 2022/9/24 11:36 上午
# @Author  : Zhiqiang He
# @Email   : tinyzqh@163.com
# @File    : ppo.py
# @Software: PyCharm


import torch
from torch import Tensor

from XuanJing.utils.torch_utils import to_torch
import torch.nn.functional as F


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PPO(object):
    def __init__(
            self,
            actor,
            optim,
            args
    ):
        """基于采样样本来更新actor的参数，critic当作是一种trick来更好更新policy而已。"""
        self.actor_net = actor.actor_net
        # TODO critic可以依据actor中的参数来进行配置，critic也需要封装到另外的一个模块去。
        self.critic_net = ValueNet(state_dim=4, hidden_dim=128)
        self.actor_optim = optim
        self.critic_optim = torch.optim.Adam(self.critic_net.parameters(), lr=1e-2)
        self.args = args
        self.learn_step = 0
        self.criterion = torch.nn.SmoothL1Loss()
        self.logging = {}

    def compute_advantage(self, gamma, lmbda, td_delta):
        td_delta = td_delta.detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = gamma * lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.tensor(advantage_list, dtype=torch.float)

    def updata_parameter(
            self,
            train_data
    ):
        obs = to_torch(train_data.get_value("obs"))
        actions = to_torch(train_data.get_value("output")["act"]).view(-1, 1)
        next_obs = to_torch(train_data.get_value("next_obs"))
        rewards = to_torch(train_data.get_value("reward")).view(-1, 1)
        done = to_torch(train_data.get_value("done")).view(-1, 1)

        # TD
        td_target = rewards + self.args.gamma * self.critic_net(next_obs) * (1 - done)
        td_delta = td_target - self.critic_net(obs)

        advantage = self.compute_advantage(self.args.gamma, 0.95, td_delta)
        old_log_probs = torch.log(F.softmax(self.actor_net(obs), dim=1).gather(1, actions)).detach()

        actor_losses = 0.0
        critic_losses = 0.0
        for i in range(self.args.update_times):
            predict_prob = self.actor_net(obs)
            softmax_probs = F.softmax(predict_prob, dim=1)
            log_prob = torch.log(softmax_probs).gather(1, actions)

            ratio = torch.exp(log_prob - old_log_probs)
            surrogate1 = ratio * advantage
            surrogate2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * advantage

            actor_loss = -torch.min(surrogate1, surrogate2).mean()
            # critic_loss = self.criterion(self.critic_net(obs), td_target.detach())
            critic_loss = F.mse_loss(self.critic_net(obs), td_target.detach()).mean()

            self.optimizer_update(self.actor_optim, actor_loss)
            self.optimizer_update(self.critic_optim, critic_loss)

            actor_losses += actor_loss.item()
            critic_losses += critic_loss.item()

        # anything you want to recorder!
        self.logging.update({
            "Learn/actor_losses": actor_losses / self.args.update_times,
            "Learn/critic_losses": critic_losses / self.args.update_times
        })

    @staticmethod
    def optimizer_update(optimizer, loss: Tensor):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()