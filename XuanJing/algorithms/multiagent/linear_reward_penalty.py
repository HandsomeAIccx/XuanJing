# -*- coding: utf-8 -*-
# @Time    : 2022/9/28 11:38 下午
# @Author  : Zhiqiang He
# @Email   : tinyzqh@163.com
# @File    : linear_reward_penalty.py
# @Software: PyCharm

import numpy as np

learning_rate1 = 0.001
learning_rate2 = 0.0005

GameMatrix = np.array([
    [0, 2, -1],
    [-1, 0, 1],
    [1, -1, 0]
])

row_action_num = GameMatrix.shape[0]
col_action_num = GameMatrix.shape[1]
m = GameMatrix.shape[0]  # number of actions

row_policy = np.ones(row_action_num) / row_action_num
col_policy = np.ones(col_action_num) / col_action_num
policies = [row_policy, col_policy]

# def get_payoff(payoff_matrix, actions, policies):
#     col_r = (policies[0] @ payoff_matrix)[actions[1]]
#     row_r = (payoff_matrix @ policies[1].T)[actions[0]]
#     return [row_r, col_r]

def get_payoff(payoff_matrix, actions):
    r = payoff_matrix[tuple(actions)]
    return [r, -r]  # [row, col]

for i in range(50000):
    row_a = np.argmax(np.random.multinomial(1, row_policy))
    col_a = np.argmax(np.random.multinomial(1, col_policy))
    actions = [row_a, col_a]
    # payoffs = get_payoff(GameMatrix, [row_a, col_a], policies)
    payoffs = get_payoff(GameMatrix, [row_a, col_a])

    for k, (a, r) in enumerate(zip(actions, payoffs)):  # iterate over agents
        for j in range(policies[k].shape[0]):  # iterate over all actions
            if j == a:
                policies[k][j] = policies[k][j] + learning_rate1 * r * (1 - policies[k][j]) - learning_rate2 * (1 - r) * \
                                 policies[k][j]
            else:
                policies[k][j] = policies[k][j] - learning_rate1 * r * policies[k][j] + learning_rate2 * (1 - r) * (
                            1. / (m - 1.) - policies[k][j])

        # above is unnormalized, normalize it to be distribution
        abs_policy = np.abs(policies[k])
        policies[k] = abs_policy / np.sum(abs_policy)
    [row_policy, col_policy] = policies

print(f'For row player, strategy is {policies[0]}')
print(f'For column player, strategy is {policies[1]}')

"""
采用Fictitious Play的方式
"""

row_value = np.zeros(GameMatrix.shape[0])
col_value = np.zeros(GameMatrix.shape[1])
min_id_list, max_id_list = [], []

for i in range(50000):
    # current l is row
    max_id = np.argmax(row_value)  # row player is maximizer
    max_id_list.append(max_id)
    l = GameMatrix[max_id]  # l is column now
    col_value += np.array(l)

    # current l is column
    min_id = np.argmin(col_value)  # column player is minimizer
    min_id_list.append(min_id)
    l = GameMatrix[:, min_id]  # l is row now
    row_value += np.array(l)

# The statistical frequencies of occurrence of different entries are just their probability masses in final policy,
# which is the average best response over history.
# hist, _ = np.histogram(max_id_list, bins=GameMatrix.shape[0])  # np.histogram is the wrong function to use!
hist = np.bincount(max_id_list, minlength=GameMatrix.shape[0])
max_policy = hist / np.sum(hist)
hist = np.bincount(min_id_list, minlength=GameMatrix.shape[1])
min_policy = hist / np.sum(hist)

row_value = row_value / (i + 1)  # historical average
col_value = col_value / (i + 1)
print(f'For row player, strategy is {max_policy}, game value (lower bound): {row_value}')
print(f'For column player, strategy is {min_policy}, game value (upper bound): {col_value}')