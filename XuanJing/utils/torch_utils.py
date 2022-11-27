# -*- coding: utf-8 -*-
# @Time    : 2022/9/24 9:19 上午
# @Author  : Zhiqiang He
# @Email   : tinyzqh@163.com
# @File    : torch_utils.py
# @Software: PyCharm

import numpy as np
import torch


def tensorify(data):
    if isinstance(data, dict):
        return {k: tensorify(v) for k, v in data.items()}
    elif isinstance(data, np.ndarray):
        if data.dtype in [np.int64, int]:
            return torch.tensor(data, dtype=torch.int64)
        else:
            return torch.tensor(data, dtype=torch.float)
    elif isinstance(data, list):
        return [torch.tensor(i) for i in data]
    else:
        return ValueError(f"Not Support type {type(data)}")


def numpyify(data):
    if isinstance(data, dict):
        return {k: numpyify(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [numpyify(i) for i in data]
    elif isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    else:
        return np.asarray(data)