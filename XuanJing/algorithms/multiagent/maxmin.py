import argparse
import numpy as np
import scipy.optimize


def solve_maxmin(A):
    """
    # we need to compile the coefficient arrays correctly for
    # scipy.optimize.linprog
    """
    n, m = A.T.shape  # we have m + 1 vars (strategies + the maxmin value)

    # objective function, negative because linprog solves a minimization problem
    c = np.zeros(m + 1)
    c[-1] = -1.

    # constraints
    # sign is changed to get an upper bound for the inequality constraints
    A_ub = np.append(- A.T, np.ones((n, 1)), 1)
    b_ub = np.zeros(n)

    # 1 ^ T x = 1 (but we need the trailing zero for v)
    A_eq = np.append(np.ones((1, m)), np.zeros((1, 1)), 1)
    b_eq = np.ones((1))

    # strategy variables are non-negative, v is free
    bounds = [(0, None) for i in range(m)] + [(None, None)]

    result = scipy.optimize.linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method='highs-ipm')

    return result.x[0:m], result.x[m]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--rows', type=int, default=100, help='The rows of the random game matrix')
    parser.add_argument('-c', '--cols', type=int, default=100, help='The columns of the random game matrix')
    parser.add_argument('-s', '--seed', type=int, default=1, help='The RNG seed')

    args = parser.parse_args()

    np.set_printoptions(precision=5)

    rng = np.random.default_rng(args.seed)
    A = rng.uniform(0, 1, (args.rows, args.cols))
    print('Computing exact game value with LP... ', end='')
    x_row_gt, v_row_gt = solve_maxmin(A)

    print(f'Game value is {v_row_gt}')