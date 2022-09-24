# -*- coding: utf-8 -*-
# @Time    : 2022/9/24 9:19 上午
# @Author  : Zhiqiang He
# @Email   : tinyzqh@163.com
# @File    : torch_utils.py
# @Software: PyCharm

import numpy as np
import torch


def to_torch(data):
    if isinstance(data, dict):
        return {k: to_torch(v) for k, v in data.items()}
    elif isinstance(data, np.ndarray):
        if data.dtype == int:
            return torch.tensor(data, dtype=torch.int64)
        else:
            return torch.tensor(data, dtype=torch.float)
    else:
        return ValueError(f"Not Support type {type(data)}")