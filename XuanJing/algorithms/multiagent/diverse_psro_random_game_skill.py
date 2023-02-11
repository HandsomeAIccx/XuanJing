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


def joint_loss(pop, payoffs, meta_nash, k, lambda_weight, lr):
    dim = payoffs.shape[0]

    br = np.zeros((dim,))
    values = []
    cards = []
    for i in range(dim):
        br_tmp = np.zeros((dim, ))
        br_tmp[i] = 1.

        aggregated_enemy = meta_nash @ pop[:k]
        value = br_tmp @ payoffs @ aggregated_enemy.T

        pop_k = lr * br_tmp + (1 - lr) * pop[k]
        pop_tmp = np.vstack((pop[:k], pop_k))
        M = pop_tmp @ payoffs @ pop_tmp.T
        metanash_tmp, _ = fictitious_play(payoffs=M, iters=1000)
        # L = np.diag(metanash_tmp[-1]) @ M @ M.T @ np.diag(metanash_tmp[-1])
        L = M @ M.T
        l_card = np.trace(np.eye(L.shape[0]) - np.linalg.inv(L + np.eye(L.shape[0])))

        cards.append(l_card)
        values.append(value)

    if np.random.randn() < lambda_weight:
        br[np.argmax(values)] = 1
    else:
        br[np.argmax(cards)] = 1

    return br


def diverse_psro_step(payoffs=None, seed=None, num_learners=None, iters=None, lr=None, improvement_pct_threshold=None):
    dim = payoffs.shape[0]
    r = np.random.RandomState(seed)
    pop = r.uniform(0, 1, (1 + num_learners, dim))  # 随机产生种群
    pop = pop / pop.sum(axis=1)[:, None]
    exploitability = get_exploitability(pop, payoffs, iters=1000)
    exps = [exploitability]
    learner_performances = [[.1] for i in range(num_learners + 1)]
    for i in range(iters):
        print(i)
        # Define the weighting towards diversity as a function of the fixed population size, this is currently a hyperparameter
        lambda_weight = 0.85
        if i % 5 == 0:
            print('iteration: ', i, ' exp full: ', exps[-1])
            print('size of pop: ', pop.shape[0])
        for j in range(num_learners):
            # first learner (when j=num_learners-1) plays against normal meta Nash
            # second learner plays against meta Nash with first learner included, etc.
            k = pop.shape[0] - j - 1
            empirical_game_matrix = pop[:k] @ payoffs @ pop[:k].T
            meta_nash, _ = fictitious_play(iters=1000, payoffs=empirical_game_matrix)
            population_strategy = meta_nash[-1] @ pop[:k]  # aggregated enemy according to nash
            best_respond = joint_loss(pop, payoffs, meta_nash[-1], k, lambda_weight, lr)  # different from psro
            pop[k] = lr * best_respond + (1 - lr) * pop[k]
            performance = pop[k] @ payoffs @ population_strategy.T + 1  # make it positive for pct calculation
            learner_performances[k].append(performance)
            # if the first learner plateaus, add a new policy to the population
            if j == num_learners - 1 and performance / learner_performances[k][-2] - 1 < improvement_pct_threshold:
                learner = np.random.uniform(0, 1, (1, dim))
                learner = learner / learner.sum(axis=1)[:, None]
                pop = np.vstack((pop, learner))
                learner_performances.append([0.1])
        # calculate exploitability for meta Nash of whole population
        exp = get_exploitability(pop, payoffs, iters=1000)
        exps.append(exp)

    def plot_error(data, label=''):
        data_mean = np.mean(np.array(data), axis=0)
        error_bars = stats.sem(np.array(data))
        plt.plot(data_mean, label=label)
        plt.fill_between([i for i in range(data_mean.size)],
                         np.squeeze(data_mean - error_bars),
                         np.squeeze(data_mean + error_bars), alpha=0.4)

    plot_error([exps], label='Diverse-PSRO')
    plt.show()


def main():
    seed = 0
    dim = 1000
    np.random.seed(seed)
    W = np.random.randn(dim, dim)
    S = np.random.randn(dim, 1)
    payoffs = 0.5 * (W - W.T) + S - S.T
    payoffs /= np.abs(payoffs).max()

    diverse_psro_step(payoffs=payoffs, seed=seed+1, num_learners=1, iters=200, lr=0.5, improvement_pct_threshold=0.03)

    print()


if __name__ == "__main__":
    main()

