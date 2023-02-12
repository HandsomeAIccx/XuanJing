import argparse
import numpy as np

from XuanJing.algorithms.multiagent.maxmin import solve_maxmin
from XuanJing.algorithms.multiagent.minmax import solve_minmax


def double_oracle(A: np.ndarray):
    """
    # solve a 2-player zero-sum game with the double-oracle algorithm
    # tabular form, matrix A is the payoff of the row player
    """
    rows = A.shape[0]
    cols = A.shape[1]

    # initialize arrays of row/column flags (if true, then the corresponding strategy is in the population)
    row_flags = [True] + (rows - 1) * [False]
    col_flags = [True] + (cols - 1) * [False]

    # initialize lists of available strategies
    row_strategies = [i for i in range(rows) if row_flags[i]]
    col_strategies = [i for i in range(cols) if col_flags[i]]

    n = 0
    while True:
        n = n + 1

        # Step 1: solve restricted game 得到收益矩阵子集
        Ar = A[np.ix_(row_strategies, col_strategies)]
        xr_row, v_row = solve_maxmin(Ar)  # 得到子博弈问题中行玩家的纳什策略
        xr_col, v_col = solve_minmax(Ar)  # 得到子博弈问题中列玩家的纳什策略

        # extend restricted row strategy
        assert len(xr_row) == len(row_strategies), "子博弈问题中行玩家的解需要与行玩家可选策略长度相同."
        x_row = np.zeros(A.shape[0])  # 初始化一个全0的原问题的行玩家的解
        for i in range(len(xr_row)):  # 将子博弈问题中的行玩家的解赋值给原问题的解
            x_row[row_strategies[i]] = xr_row[i]

        # extend restricted col strategy
        assert len(xr_col) == len(col_strategies), "子博弈问题中列玩家的解需要与列玩家可选策略长度相同."
        x_col = np.zeros(A.shape[1])  # 初始化一个全0的列玩家的解
        for i in range(len(xr_col)):  # 将子博弈问题中的列玩家的解赋值给原问题
            x_col[col_strategies[i]] = xr_col[i]

        # Step 2: compute response values for the restricted strategies
        row_values = A @ x_col  # 行玩家的收益矩阵
        col_values = A.T @ x_row  # 列玩家的收益矩阵

        updated = False

        vr = row_values.max()  # 行玩家收益矩阵中的最大一项, 也就是Best Respond.
        for i in range(len(row_values)):  # 找到行玩家的原问题Best Respond, 并将其添加到下一次的可选策略中去
            if np.isclose(row_values[i], vr) and row_flags[i] is False:
                row_strategies.append(i)
                row_flags[i] = True
                updated = True
                break

        # min val for the column player
        vc = col_values.min()  # 列玩家收益矩阵中的最大一项, 也就是Best Respond
        for i in range(len(col_values)):  # 找到列玩家的原问题Best Respond, 并将其添加到下一次的可选策略中去
            if np.isclose(col_values[i], vc) and col_flags[i] is False:
                col_strategies.append(i)
                col_flags[i] = True
                updated = True
                break

        if not updated:
            return x_row, x_col, vr, vc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--rows', type=int, default=100, help='The rows of the random game matrix')
    parser.add_argument('-c', '--cols', type=int, default=100, help='The columns of the random game matrix')
    parser.add_argument('-s', '--seed', type=int, default=1, help='The RNG seed')

    args = parser.parse_args()

    np.set_printoptions(precision=5)

    rng = np.random.default_rng(args.seed)
    A = rng.uniform(0, 1, (args.rows, args.cols))
    print('Solving with Double Oracle...')
    x_row_do, x_col_do, v_row_do, v_col_do = double_oracle(A)
    print(f'Row value = {v_row_do}, column value = {v_col_do}, gap = {v_row_do - v_col_do}')