# -*- coding: utf-8 -*-
# @Time    : 2022/7/16 12:00 下午
# @Author  : Zhiqiang He
# @Email   : tinyzqh@163.com
# @File    : base.py
# @Software: PyCharm


class BaseActor(object):
    """Sampling Data Based on Given Actor and Env

    """

    def __init__(
            self,
            actor,
            env,
            buffer
    ) -> None:
        super(BaseActor, self).__init__()

    pass