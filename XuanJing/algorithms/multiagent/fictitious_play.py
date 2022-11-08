# -*- coding: utf-8 -*-
# @Time    : 2022/9/27 8:41 下午
# @Author  : Zhiqiang He
# @Email   : tinyzqh@163.com
# @File    : fictitious_play.py
# @Software: PyCharm

"""
Fictitious Play是找出对手的最佳反应策略
"""

import numpy as np

GameMatrix = np.array([
    [-2, 3],
    [3, -4]
])

index = 0
p2_payoff = np.zeros((1, 2))
p1_payoff = np.zeros((2, 1))

for i in range(100000):
    p1_payoff += GameMatrix[:, index][:, np.newaxis]
    index = np.argmax(p1_payoff)
    upper_policy = p1_payoff[index, :] / (i + 1)

    p2_payoff += GameMatrix[index, :][np.newaxis, :]
    index = np.argmin(p2_payoff)
    low_policy = p2_payoff[:, index] / (i + 1)

print("lower", low_policy)
print("upper", upper_policy)


# case 2

# GameMatrix = np.array([[2,1], [1,1]])
GameMatrix = np.array([[0,2,-1], [-1,0,1], [1,-1,0]])

Itr = 10000


def fp(GameMatrix):
    """ Fictitious play on normal-from game."""
    # a random initial step on row
    row_value = np.zeros(GameMatrix.shape[0])
    col_value = np.zeros(GameMatrix.shape[1])
    min_id_list, max_id_list = [], []

    for i in range(Itr):
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

    row_value = row_value / (i + 1)  # historical average, row value is lower bound
    col_value = col_value / (i + 1)  # historical average, column value is upper bound
    print(f'For row player, strategy is {max_policy}, game value (lower bound): {row_value}')
    print(f'For column player, strategy is {min_policy}, game value (upper bound): {col_value}')

    return (max_policy, min_policy), (row_value, col_value)


fp(GameMatrix)

if __name__ == "__main__":
    pass