# -*- coding: utf-8 -*-
# @Time    : 2022/9/25 6:22 下午
# @Author  : Zhiqiang He
# @Email   : tinyzqh@163.com
# @File    : gail.py
# @Software: PyCharm


class GAIL(object):
    def __init__(self):
        self.actor_net = None
        self.discriminator = None
        self.optim = None

    def updata_parameter(
            self,
            train_data
    ):
        pass