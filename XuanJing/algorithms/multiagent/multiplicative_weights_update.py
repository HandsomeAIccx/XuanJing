# -*- coding: utf-8 -*-
# @Time    : 2022/9/28 11:58 下午
# @Author  : Zhiqiang He
# @Email   : tinyzqh@163.com
# @File    : multiplicative_weights_update.py
# @Software: PyCharm

import copy, time

import numpy as np
# ! pip install pandas

GameMatrix = np.array([[0,2,-1], [-1,0,1], [1,-1,0]])
Itr = 1000

row_action_num = GameMatrix.shape[0]
col_action_num = GameMatrix.shape[1]

learning_rate = np.sqrt(np.log(row_action_num)/Itr)  # sqrt(log |A| / T)
print('learning rate: ', learning_rate)

row_policy = np.ones(row_action_num)/row_action_num
col_policy = np.ones(col_action_num)/col_action_num
policies = np.array([row_policy, col_policy])
final_policy = copy.deepcopy(policies)
print(f'initial strategies, row: {row_policy}, column: {col_policy}')

def get_payoff_vector(payoff_matrix, opponent_policy):
    payoff = opponent_policy @ payoff_matrix
    return payoff

# payoff = get_payoff(GameMatrix, [0,1])
# print(payoff)

for i in range(Itr):
    policies_ = copy.deepcopy(policies)  # track old value before update (update is inplace)
    # for row player, maximizer
    payoff_vec = get_payoff_vector(GameMatrix.T, policies_[1])
    policies[0] = policies[0] * np.exp(learning_rate*payoff_vec)

    # for col player, minimizer
    payoff_vec = get_payoff_vector(GameMatrix, policies_[0])
    policies[1] = policies[1] * np.exp(-learning_rate*payoff_vec)

    # above is unnormalized, normalize it to be distribution
    # for k in range(len(policies)):
    #     abs_policy = np.abs(policies[k])
    #     policies[k] = abs_policy/np.sum(abs_policy)
    policies = policies/np.expand_dims(np.sum(policies, axis=-1), -1)

    # MWU is average-iterate coverging, so accumulate polices
    final_policy += policies

final_policy = final_policy / (Itr+1)


print(f'For row player, strategy is {final_policy[0]}')
print(f'For column player, strategy is {final_policy[1]}')