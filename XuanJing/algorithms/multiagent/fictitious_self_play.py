import numpy as np

eta = 0.2


def sample_from_categorical(dist):
    dist = [i / sum(dist) for i in dist]
    return np.argmax(np.random.multinomial(1, dist))


def matrix_fictitious_self_play(matrix):
    """
    Best Respond的求解并不容易, 因此需要放宽Best Respond的求解精度.
    1. Origin FSP:
    Supervise Learning for achieving the average policy.  # 基于自己的历史信息获取 average policy.
    Reinforcement Learning for reaching the best respond policy. 基于对手的历史信息, 获取Best Respond.
    2. Non-Neural Version FSP:
    """
    dim = matrix.shape[0]
    row_player_strategy = np.random.uniform(0, 1, (dim, 1))
    col_player_strategy = np.random.uniform(0, 1, (1, dim))
    for i in range(10000):
        # for row player
        average = np.average(col_player_strategy, axis=0)  # Step 1: 获取对手的历史平均策略.
        best_respond = np.zeros((dim, 1))
        if sample_from_categorical([1 - eta, eta]):  # Step 2.1: 基于对手的历史平均策得到最优响应.
            best_respond[np.argmax(matrix @ average.T)] = 1
        else:  # Step 2.2: 采样得到
            idx = sample_from_categorical(np.average(col_player_strategy, axis=0))
            best_respond[np.argmax(matrix[:, idx])] = 1
        row_player_strategy = np.hstack((row_player_strategy, best_respond))
        # for col player
        average = np.average(row_player_strategy, axis=1)
        best_respond = np.zeros((1, dim))
        if sample_from_categorical([1 - eta, eta]):  # Step 2.1: 基于对手的历史平均策得到最优响应.
            best_respond[:, np.argmin(average.T @ matrix)] = 1
        else:  # Step 2.2: 采样得到
            idx = sample_from_categorical(np.average(row_player_strategy, axis=1))
            best_respond[:, np.argmax(matrix[idx, :])] = 1

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
    matrix_fictitious_self_play(matrix_instance)






