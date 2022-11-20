# -*- coding: utf-8 -*-
# @Time    : 2022/7/16 12:00 下午
# @Author  : Zhiqiang He
# @Email   : tinyzqh@163.com
# @File    : base.py
# @Software: PyCharm

class BaseLearner(object):
    @staticmethod
    def save_logging(writer, log_data, step):
        assert isinstance(log_data, dict), f"input type {type(log_data)} is not a dict!"
        for k, v in log_data.items():
            writer.add_scalar(k, v, step)

