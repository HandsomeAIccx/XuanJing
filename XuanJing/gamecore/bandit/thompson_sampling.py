import numpy as np
from XuanJing.gamecore.bandit.bernoulli_bandit_env import BernoulliBanditEnv


class ThompsonSamplingAgent(object):
    """
    paper: An Empirical Evaluation of Thompson Sampling
    link: https://proceedings.neurips.cc/paper/2011/file/e53a0a2978c28872a4505bdb51db06dc-Paper.pdf
    """
    def __init__(self, env, epsilon=0.01, init_prob=1.0):
        super(ThompsonSamplingAgent, self).__init__()
        self.env = env
        self.counts = np.zeros(self.env.K)
        self.regret = 0.0
        self.regret_history = []
        self.estimates = np.array([init_prob] * self.env.K)
        self.total_cnt = 0
        self._a = np.ones(self.env.K)  # 列表，表示每根拉杆奖励为1的次数。
        self._b = np.ones(self.env.K)  # 列表，表示每根拉杆奖励为0的次数。

    def take_action(self):
        samples = np.random.beta(self._a, self._b)
        k = np.argmax(samples)
        return k

    def update(self, k, r):
        self._a[k] += r
        self._b[k] += (1 - r)
        self.counts[k] += 1
        self.update_regret(k)

    def update_regret(self, k):
        self.regret += self.env.best_prob - self.env.probs[k]
        self.regret_history.append(self.regret)


if __name__ == "__main__":
    np.random.seed(0)
    env = BernoulliBanditEnv(10)
    agent = ThompsonSamplingAgent(env)
    for i in range(5000):
        a = agent.take_action()
        r = env.step(a)
        agent.update(a, r)

    import matplotlib.pylab as plt
    plt.plot(agent.regret_history)
    plt.show()