# -*- coding: utf-8 -*-
# @Time    : 2022/7/16 1:41 下午
# @Author  : Zhiqiang He
# @Email   : tinyzqh@163.com
# @File    : vecbase.py
# @Software: PyCharm
import gym
import numpy as np
from abc import ABC, abstractmethod


class BaseVectorEnv(ABC):
    def __init__(self, env_fns, reset_after_done):
        self._env_fns = env_fns
        self.env_num = len(env_fns)
        self._reset_after_done = reset_after_done

    def is_reset_after_done(self):
        return self._reset_after_done

    def __len__(self):
        return self.env_num

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def seed(self, seed=None):
        pass

    @abstractmethod
    def render(self):
        pass

    @abstractmethod
    def close(self):
        pass


class VectorEnv(object):
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
        self.envs = [env_fun() for env_fun in env_fns]
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def reset(self):
        return np.stack([e.reset() for e in self.envs])

    def step(
            self,
            action
    ):
        assert len(action) == len(self.envs), "Env Num Error!"
        result = zip(*[e.step(a) for e, a in zip(self.envs, action)])
        obs, reward, done, info = result
        return map(np.stack, [obs, reward, done, info])

    def seed(self):
        return [self._env_fns[id].seed() for id in range(self._env_num)]

    def render(self):
        pass

    def close(self):
        pass


if __name__ == "__main__":
    env_num = 6
    envs = VectorEnv([lambda: gym.make('CartPole-v1') for _ in range(env_num)])
    obs = envs.reset()
    obs, rew, done, info = envs.step([1] * env_num)
    envs.close()
