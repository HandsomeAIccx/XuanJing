import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def get_best_respond(strategy, payoffs):
    """
    给定策略和PayOffs矩阵, 返回Best Respond.
    """
    row_weighted_payouts = strategy @ payoffs
    best_respond = np.zeros_like(row_weighted_payouts)
    best_respond[np.argmin(row_weighted_payouts)] = 1
    return best_respond


def fictitious_play(iters=2000, payoffs=None):
    """
    Fictitious play as a nash equilibrium solver.
    """
    dim = payoffs.shape[0]
    population = np.random.uniform(0, 1, (1, dim))  # 初始种群策略
    population = population / population.sum(axis=1)[:, None]
    averages = population
    exps = []
    for i in range(iters):
        average = np.average(population, axis=0)  # Step 1: 历史平均策略
        best_respond = get_best_respond(average, payoffs)  # Step 2: Best Respond求解.
        exp1 = average @ payoffs @ best_respond.T
        exp2 = best_respond @ payoffs @ average.T
        exps.append(exp2 - exp1)
        averages = np.vstack((averages, average))
        population = np.vstack((population, best_respond))  # Step 3: 策略扩张
    return averages, exps


def get_exploitability(population, payoffs, iters=None):
    """
    Solve exploitability of a nash equilibrium over a fixed population.
    """
    empirical_game_matrix = population @ payoffs @ population.T
    averages, exps = fictitious_play(iters=iters, payoffs=empirical_game_matrix)  # 得到empirical game下的均衡策略.
    strategy = averages[-1] @ population  # 得到population下的混合策略
    best_respond = get_best_respond(strategy, payoffs=payoffs)
    exp1 = strategy @ payoffs @ best_respond.T
    exp2 = best_respond @ payoffs @ strategy
    return exp2 - exp1


def self_play_step(payoffs=None, seed=None, num_learners=None, iters=None, lr=None, improvement_pct_threshold=None):
    dim = payoffs.shape[0]
    r = np.random.RandomState(seed)
    pop = r.uniform(0, 1, (1 + num_learners, dim))  # 随机产生种群
    pop = pop / pop.sum(axis=1)[:, None]
    exploitability = get_exploitability(pop, payoffs, iters=1000)
    exps = [exploitability]
    performances = [0.01]
    for i in range(iters):
        if i % 10 == 0:
            print('iteration: ', i, 'exploitability: ', exps[-1])

            best_respond = get_best_respond(pop[-2], payoffs=payoffs)  # 获取对手策略下的best respond.
            pop[-1] = lr * best_respond + (1 - lr) * pop[-1]  # 基于best respond更新自己的平均策略
            performance = pop[-1] @ payoffs @ pop[-2].T + 1
            performances.append(performance)
            if performance / performances[-2] - 1 < improvement_pct_threshold:
                learner = np.random.uniform(0, 1, (1, dim))
                learner = learner / learner.sum(axis=1)[:, None]
                pop = np.vstack((pop, learner))
            exp = get_exploitability(pop, payoffs, iters=1000)
            exps.append(exp)

    def plot_error(data, label=''):
        data_mean = np.mean(np.array(data), axis=0)
        error_bars = stats.sem(np.array(data))
        plt.plot(data_mean, label=label)
        plt.fill_between([i for i in range(data_mean.size)],
                         np.squeeze(data_mean - error_bars),
                         np.squeeze(data_mean + error_bars), alpha=0.4)

    plot_error([exps], label='Self-Play')
    plt.show()


def main():
    seed = 0
    dim = 1000
    np.random.seed(seed)
    W = np.random.randn(dim, dim)
    S = np.random.randn(dim, 1)
    payoffs = 0.5 * (W - W.T) + S - S.T
    payoffs /= np.abs(payoffs).max()

    self_play_step(payoffs=payoffs, seed=seed+1, num_learners=1, iters=200, lr=0.5, improvement_pct_threshold=0.03)

    print()


if __name__ == "__main__":
    main()

