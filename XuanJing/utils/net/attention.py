# -*- coding: utf-8 -*-
# @Time    : 2022/10/7 6:27 下午
# @Author  : Zhiqiang He
# @Email   : tinyzqh@163.com
# @File    : attention.py
# @Software: PyCharm


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        pass

    def forward(self, query, key, value=None, mask=None):
        """
        query: [B, target_len, feature]
        key: [B, seq_len, feature]
        """
        # d_k = query.size(-1)
        score = torch.matmul(query, key.transpose(-2, -1)) # / math.sqrt(d_k)

        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)

        # p_attn = F.softmax(score.squeeze(1), dim=-1)
        return score.squeeze(1)


