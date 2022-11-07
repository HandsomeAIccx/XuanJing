# -*- coding: utf-8 -*-
# @Time    : 2022/9/24 11:42 上午
# @Author  : Zhiqiang He
# @Email   : tinyzqh@163.com
# @File    : advantage.py
# @Software: PyCharm


from XuanJing.env.sample.patch import Patch
import numpy as np

def enhance_advantage(patch_data):
    assert isinstance(patch_data, Patch)
    # obs = patch_data.__dict__['obs']
    # patch_data.__dict__['next_obs']
    return patch_data