# -*- coding: utf-8 -*-
# @Time    : 2022/10/3 7:37 下午
# @Author  : Zhiqiang He
# @Email   : tinyzqh@163.com
# @File    : combinatorial_rl.py
# @Software: PyCharm


import torch
import torch.nn as nn


class CombinatorialRL(nn.Module):
    def __init__(self, actor, optim, args):
        super(CombinatorialRL, self).__init__()
        self.args = args
        self.actor_net = actor
        self.actor_optim = optim
        self.critic_exp_mvg_avg = None

    def updata_parameter(
            self,
            train_data
    ):
        probs, actions, actions_idxs = self.actor_net(train_data)
        R = self.tsp_reward(actions)
        if self.critic_exp_mvg_avg == None:
            critic_exp_mvg_avg = R.mean()
        else:
            critic_exp_mvg_avg = (self.critic_exp_mvg_avg * self.args.beta) + ((1. - self.args.beta) * R.mean())

        advantage = R - critic_exp_mvg_avg

        logprobs = 0
        for prob in probs:
            logprob = torch.log(prob)
            logprobs += logprob
        logprobs[logprobs < -1000] = 0.

        actor_loss = (advantage * -logprobs).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), float(self.args.max_grad_norm), norm_type=2)

        self.actor_optim.step()

        self.critic_exp_mvg_avg = critic_exp_mvg_avg.detach()

        return R

    def tsp_reward(self, sample_solution):
        n = len(sample_solution)
        tour_len = torch.zeros([self.args.batch_size])

        for i in range(n - 1):
            tour_len += torch.norm(sample_solution[i] - sample_solution[i + 1], dim=1)
        tour_len += torch.norm(sample_solution[n - 1] - sample_solution[0], dim=1)

        return -tour_len
