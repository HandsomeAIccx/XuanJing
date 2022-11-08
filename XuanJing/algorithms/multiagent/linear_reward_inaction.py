# -*- coding: utf-8 -*-
# @Time    : 2022/9/27 10:46 下午
# @Author  : Zhiqiang He
# @Email   : tinyzqh@163.com
# @File    : linear_reward_inaction.py
# @Software: PyCharm


import numpy as np
import pandas as pd

learning_rate = 0.01

GameMatrix = np.array([
    [2, 1],
    [1, 1]
])

row_action_num = GameMatrix.shape[0]
col_action_num = GameMatrix.shape[1]

row_policy = np.ones(row_action_num) / row_action_num
col_policy = np.ones(col_action_num) / col_action_num

policies = [row_policy, col_policy]

for i in range(100000):
    row_a = np.argmax(np.random.multinomial(1, row_policy))
    col_a = np.argmax(np.random.multinomial(1, col_policy))
    actions = [row_a, col_a]

    r = GameMatrix[tuple(actions)]
    payoffs = [r, -r]

    for k, (a, r) in enumerate(zip(actions, payoffs)): # 对智能体迭代
        for j in range(policies[k].shape[0]): # 对动作迭代
            if j == a:
                policies[k][j] = policies[k][j] + learning_rate * r * (1 - policies[k][j])
            else:
                policies[k][j] = policies[k][j] - learning_rate * r * policies[k][j]

            abs_policy = np.abs(policies[k])
            policies[k] = abs_policy / np.sum(abs_policy)

    [row_policy, col_policy] = policies

print(f'For row player, strategy is {policies[0]}')
print(f'For column player, strategy is {policies[1]}')

# Fictitious Play

"""
采用Fictitious Play的方式
"""

row_value = np.zeros(GameMatrix.shape[0])
col_value = np.zeros(GameMatrix.shape[1])
min_id_list, max_id_list = [], []

for i in range(10000):
    max_id = np.argmax(row_value)
    max_id_list.append(max_id)
    l = GameMatrix[max_id]
    col_value += np.array(l)

    min_id = np.argmin(col_value)
    min_id_list.append(min_id)
    l = GameMatrix[:, min_id]
    row_value += np.array(l)

hist = np.bincount(max_id_list, minlength=GameMatrix.shape[0])
max_policy = hist / np.sum(hist)
hist = np.bincount(min_id_list, minlength=GameMatrix.shape[1])
min_policy = hist / np.sum(hist)

row_value = row_value / (i + 1)
col_value = col_value / (i + 1)

print(f'For row player, strategy is {max_policy}, game value (lower bound): {row_value}')
print(f'For column player, strategy is {min_policy}, game value (upper bound): {col_value}')
