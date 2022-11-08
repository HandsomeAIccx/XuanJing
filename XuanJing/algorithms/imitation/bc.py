# -*- coding: utf-8 -*-
# @Time    : 2022/9/25 5:56 下午
# @Author  : Zhiqiang He
# @Email   : tinyzqh@163.com
# @File    : bc.py
# @Software: PyCharm

class BC(object):
    def __init__(self):
        self.actor_net = None
        self.optim = None

    def updata_parameter(
            self,
            train_data
    ):
        pass