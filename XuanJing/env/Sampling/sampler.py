# -*- coding: utf-8 -*-
# @Time    : 2022/7/16 5:16 下午
# @Author  : Zhiqiang He
# @Email   : tinyzqh@163.com
# @File    : sampler.py
# @Software: PyCharm


class Sampler(object):
    def __init__(
            self,
            actor,
            env,
            buffer,
            exploration_noise
    ):
        super(Sampler, self).__init__()
        self.actor = actor
        self.envs = env
        self.buffer = buffer
        self.reset(False)

    def reset(self, reset_buffer):
        self.data =

    def sample(
            self,
    ):



