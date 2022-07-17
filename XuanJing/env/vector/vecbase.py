# -*- coding: utf-8 -*-
# @Time    : 2022/7/16 1:41 下午
# @Author  : Zhiqiang He
# @Email   : tinyzqh@163.com
# @File    : vecbase.py
# @Software: PyCharm
import gym
import numpy as np


class BaseVectorEnv(object):
    """ Base class for vectorized environment wrapper.
    Usage:
    ::
        env_num = 6
        envs =
        # TODO 框架流程测试版本
    """

    def __init__(
            self,
            env_fns,
    ):
        self._env_fns = env_fns
        self._env_num = len(env_fns)
        assert len(env_fns) > 0, "print len of env list <= 0!"
        self.observation_space = env_fns[0].observation_space
        self.action_space = env_fns[0].action_space

    def reset(self):
        reset_list = []
        for id in range(self._env_num):
            reset_list.append(self._env_fns[id].reset())

        obs_list = [r[0] for r in reset_list]
        obs = np.stack(obs_list)
        return obs

    def step(
            self,
            action
    ):
        result = []
        for action, env_id in zip(action, range(self._env_num)):
            obs, reward, done, info = self._env_fns[env_id].step(action)
            result.append((obs, reward, done, info))

        obs_list, reward_list, done_list, info_list = zip(*result)
        obs_stack, reward_stack, done_stack, info_stack = map(np.stack, [obs_list, reward_list, done_list, info_list])
        return obs_stack, reward_stack, done_stack, info_stack

    def seed(self):
        return [self._env_fns[id].seed() for id in range(self._env_num)]

    def render(self):
        pass

    def close(self):
        pass


if __name__ == "__main__":
    env_num = 6
    envs = BaseVectorEnv([gym.make('CartPole-v1') for _ in range(env_num)])
    envs.reset()
    obs, rew, done, info = envs.step([1] * env_num)
    envs.close()
