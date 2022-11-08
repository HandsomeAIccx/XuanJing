# -*- coding: utf-8 -*-
# @Time    : 2022/9/27 8:56 下午
# @Author  : Zhiqiang He
# @Email   : tinyzqh@163.com
# @File    : fictitious_self_play.py
# @Software: PyCharm

import numpy as np

GameMatrix = np.array([[0, 2, -1], [-1, 0, 1], [1, -1, 0]])

eta = 0.2
avg_warmup = 0.2
N = 300  # 采样多次获得平均策略

row_value = np.zeros(GameMatrix.shape[0])
col_value = np.zeros(GameMatrix.shape[1])

min_id_list, max_id_list = [], []

max_policy = 1./GameMatrix.shape[0] * np.ones(GameMatrix.shape[0])
min_policy = 1./GameMatrix.shape[1] * np.ones(GameMatrix.shape[1])

def sample_from_categorical(dist):
    """
    sample once from a categorical distribution, return the entry index.
    dist: should be a list or array of probabilities for a categorical distribution
    """
    return np.argmax(np.random.multinomial(1, dist))


for i in range(1000):
    if i < avg_warmup or np.isnan(max_policy).any() or sample_from_categorical([1-eta, eta]):
        max_id = np.argmax(row_value)
        max_id_list.append(max_id)
    col_value = np.zeros(GameMatrix.shape[1])
    for j in range(N):
        max_id = sample_from_categorical(max_policy)
        l = GameMatrix[max_id]
        col_value += np.array(l)
    col_value = col_value / N

    if i < avg_warmup or np.isnan(min_policy).any() or sample_from_categorical([1-eta, eta]):
        min_id = np.argmin(col_value)
        min_id_list.append(min_id)
    row_value = np.zeros(GameMatrix.shape[0])
    for j in range(N):
        min_id = sample_from_categorical(min_policy)
        l = GameMatrix[:, min_id]
        row_value += np.array(l)
    row_value = row_value / N

    # average policy for each player
    hist = np.bincount(max_id_list, minlength=GameMatrix.shape[0])
    max_policy = hist / np.sum(hist)
    hist = np.bincount(min_id_list, minlength=GameMatrix.shape[1])
    min_policy = hist / np.sum(hist)

    row_value = row_value / (i+1)
    col_value = col_value / (i+1)

print("max policy", max_policy)
print("min policy", min_policy)

print("row value", row_value)
print("col value", col_value)









