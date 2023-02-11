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


def psro_rectified_step(payoffs=None, seed=None, num_start_strats=None, num_pseudo_learners=None, iters=None, lr=None, threshold=None):
    dim = payoffs.shape[0]
    r = np.random.RandomState(seed)
    pop = r.uniform(0, 1, (num_start_strats, dim))  # 随机产生种群
    pop = pop / pop.sum(axis=1)[:, None]
    exploitability = get_exploitability(pop, payoffs, iters=1000)
    exps = [exploitability]

    counter = 0
    eps = 0.01
    while counter < iters * num_pseudo_learners:
        if counter % (5 * num_pseudo_learners) == 0:
            print('iteration: ', int(counter / num_pseudo_learners), ' exp: ', exps[-1])
            print('size of population: ', pop.shape[0])

        new_pop = np.copy(pop)
        emp_game_matrix = pop @ payoffs @ pop.T
        averages, _ = fictitious_play(payoffs=emp_game_matrix, iters=iters)

        # go through all policies. If the policy has positive meta Nash mass,
        # find policies it wins against, and play against meta Nash weighted mixture of those policies
        for j in range(pop.shape[0]):
            if counter > iters * num_pseudo_learners:
                break
            # if positive mass, add a new learner to pop and update it with steps, submit if over thresh
            # keep track of counter
            if averages[-1][j] > eps:
                # create learner
                learner = np.random.uniform(0, 1, (1, dim))
                learner = learner / learner.sum(axis=1)[:, None]
                new_pop = np.vstack((new_pop, learner))
                idx = new_pop.shape[0] - 1

                current_performance = 0.02
                last_performance = 0.01
                while current_performance / last_performance - 1 > threshold:
                    counter += 1
                    mask = emp_game_matrix[j, :]
                    mask[mask >= 0] = 1
                    mask[mask < 0] = 0
                    weights = np.multiply(mask, averages[-1])
                    weights /= weights.sum()
                    strat = weights @ pop
                    br = get_best_respond(strat, payoffs=payoffs)
                    new_pop[idx] = lr * br + (1 - lr) * new_pop[idx]
                    last_performance = current_performance
                    current_performance = new_pop[idx] @ payoffs @ strat + 1

                    if counter % num_pseudo_learners == 0:
                        # count this as an 'iteration'
                        # exploitability
                        exp = get_exploitability(new_pop, payoffs, iters=1000)
                        exps.append(exp)

        pop = np.copy(new_pop)

    def plot_error(data, label=''):
        data_mean = np.mean(np.array(data), axis=0)
        error_bars = stats.sem(np.array(data))
        plt.plot(data_mean, label=label)
        plt.fill_between([i for i in range(data_mean.size)],
                         np.squeeze(data_mean - error_bars),
                         np.squeeze(data_mean + error_bars), alpha=0.4)

    plot_error([exps], label='Rectified-PSRO')
    plt.show()


def main():
    seed = 0
    dim = 1000
    np.random.seed(seed)
    W = np.random.randn(dim, dim)
    S = np.random.randn(dim, 1)
    payoffs = 0.5 * (W - W.T) + S - S.T
    payoffs /= np.abs(payoffs).max()

    psro_rectified_step(payoffs=payoffs, seed=seed+1, num_start_strats=1, num_pseudo_learners=2, iters=200, lr=0.5, threshold=0.03)

    print()


if __name__ == "__main__":
    main()

