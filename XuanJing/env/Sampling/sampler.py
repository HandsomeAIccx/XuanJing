# -*- coding: utf-8 -*-
# @Time    : 2022/7/16 5:16 下午
# @Author  : Zhiqiang He
# @Email   : tinyzqh@163.com
# @File    : sampler.py
# @Software: PyCharm

from XuanJing.env.Sampling.patch import Patch
import copy
import numpy as np


class Sampler(object):
    def __init__(
            self,
            actor,
            env,
            args
    ):
        super(Sampler, self).__init__()
        self.actor = actor
        self.env = env
        self.args = args
        self.patch_data = Patch()
        self.init_state = True
        self.current_obs = None
        self.episode_reward = 0
        self.episodes_reward = []
        self.episode_done = False

    def sample_episode(self, n_episode=0):
        assert n_episode > 0, "episode len must > 0!"
        cur_episode = 0
        obs = self.env.reset()
        while True:
            actor_out = self.actor.sample_forward(obs)
            action = actor_out['act']
            obs_next, reward, done, info = self.env.step(action)
            self.patch_data.add(
                Patch(
                    obs=obs,
                    output=actor_out,
                    reward=reward,
                    done=done
                )
            )
            if done:
                cur_episode += 1
                obs_next = self.env.reset()
            if cur_episode >= n_episode:
                break
            obs = obs_next

    def sample_step(self, n_step=0):
        assert n_step > 0, "n_step len must > 0!"
        cur_step = 0
        if self.current_obs is None:
            self.current_obs = self.env.reset()
        while True:
            actor_out = self.actor.sample_forward(self.current_obs)
            obs_next, reward, done, info = self.env.step(actor_out['act'])
            self.episode_reward += reward[0]
            self.patch_data.add(
                Patch(
                    obs=self.current_obs,
                    output=actor_out,
                    reward=reward,
                    done=done,
                    next_obs=obs_next
                )
            )
            if done:
                obs_next = self.env.reset()
                self.episodes_reward.append(self.episode_reward)
                self.episode_reward = 0
                self.episode_done = True
            else:
                self.episode_done = False

            self.current_obs = obs_next

            cur_step += 1
            if cur_step >= n_step:
                break


    def get_episode_result(self):
        return {"episodes_reward": self.episodes_reward}

    def get_sampler_data(self):
        patch_data = copy.copy(self.patch_data)
        self.patch_data.clear()
        return patch_data
