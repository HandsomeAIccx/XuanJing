# -*- coding: utf-8 -*-
# @Time    : 2022/8/29 9:11 下午
# @Author  : Zhiqiang He
# @Email   : tinyzqh@163.com
# @File    : test_patch.py
# @Software: PyCharm

import numpy as np
from XuanJing.env.sample.patch import Patch


if __name__ == "__main__":
    patch_instance = Patch()
    print(patch_instance.__dict__)
    patch_test = Patch(
        obs=np.array((1, 2)),
        action=np.array((1, 1)),
        reward=np.array((1, 1)),
        next_obs=np.array((1, 2)))
    patch_instance.add(patch_test)
    print(patch_instance.__dict__)
