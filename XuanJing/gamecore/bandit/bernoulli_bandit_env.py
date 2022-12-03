import numpy as np


class BernoulliBanditEnv(object):
    """
    paper title: Finite-time Analysis of the Multiarmed Bandit Problem
    link: https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf
    """
    def __init__(self, k):
        self.probs = np.random.uniform(size=k)

        self.best_idx = np.argmax(self.probs)
        self.best_prob = self.probs[self.best_idx]
        self.K = k

    def step(self, k):
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0
