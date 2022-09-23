# -*- coding: utf-8 -*-
# @Time    : 2022/8/30 9:33 下午
# @Author  : Zhiqiang He
# @Email   : tinyzqh@163.com
# @File    : next_state.py
# @Software: PyCharm

from XuanJing.env.Sampling.patch import Patch
import numpy as np

def enhance_next_state(patch_data):
    assert isinstance(patch_data, Patch)
    # obs = patch_data.__dict__['obs']
    # patch_data.__dict__['next_obs']
    return patch_data