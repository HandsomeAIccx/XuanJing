"""
Fictitious Play是找出对手的最佳反应策略
"""

import numpy as np


def matrix_fictitious_play(matrix):
    """
    每个智能体拥有两个策略集: 最优策略集, 历史平均策略集.
    Step 1: 依据对手历史平均策略集, 找到一个最优的针对策略。
    Step 2: 根据历史平均策略和本轮最优策略更新自己的历史平均策略。
    """
    dim = matrix.shape[0]
    row_player_strategy = np.random.uniform(0, 1, (dim, 1))
    col_player_strategy = np.random.uniform(0, 1, (1, dim))
    for i in range(10000):
        # for row player
        average = np.average(col_player_strategy, axis=0)  # Step 1: 获取对手的历史平均策略.
        best_respond = np.zeros((dim, 1))
        best_respond[np.argmax(matrix @ average.T)] = 1  # Step 2: 基于对手的历史平均策得到最优响应.
        row_player_strategy = np.hstack((row_player_strategy, best_respond))
        # for col player
        average = np.average(row_player_strategy, axis=1)
        best_respond = np.zeros((1, dim))
        best_respond[:, np.argmin(average.T @ matrix)] = 1
        col_player_strategy = np.vstack((col_player_strategy, best_respond))
        if i % 10 == 0:
            print("row", np.average(row_player_strategy, axis=1))
            print("col", np.average(col_player_strategy, axis=0))


if __name__ == "__main__":
    matrix_instance = np.array([
        [0, -1, 1],
        [1, 0, -1],
        [-1, 1, 0]
    ])
    matrix_fictitious_play(matrix_instance)

