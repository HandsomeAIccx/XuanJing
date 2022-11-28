import numpy as np

from XuanJing.gamecore.bandit.solver import SolverAgent
from XuanJing.gamecore.bandit.bernoulli_bandit_env import BernoulliBanditEnv


class EpsilonGreedyAgent(object):
    def __init__(self, env, epsilon=0.01, init_prob=1.0):
        super(EpsilonGreedyAgent, self).__init__()
        self.env = env
        self.counts = np.zeros(self.env.k)
        self.regret = 0.0
        self.action_history = []
        self.regret_history = []
        self.epsilon = epsilon
        self.estimates = np.array([init_prob] * self.env.k)
        self.total_cnt = 0


    def take_action(self):
        if np.random.random() < self.epsilon:
            k = np.random.randint(0, self.env.k)
        else:
            k = np.argmax(self.estimates)

        return k

    def update(self, k, r):
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        self.counts[k] += 1
        self.action_history.append(k)
        self.update_regret(k)

    def update_regret(self, k):
        self.regret += self.env.best_prob - self.env.probs[k]
        self.regret_history.append(self.regret)


if __name__ == "__main__":
    np.random.seed(1)
    env = BernoulliBanditEnv(10)
    agent = EpsilonGreedyAgent(env)
    for i in range(5000):
        a = agent.take_action()
        r = env.step(a)
        agent.update(a, r)

    import matplotlib.pylab as plt
    plt.plot(agent.regret_history)
    plt.show()