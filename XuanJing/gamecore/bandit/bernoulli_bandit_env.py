import numpy as np


class BernoulliBanditEnv(object):
    def __init__(self, k):
        np.random.seed(1)
        self.probs = np.random.uniform(size=k)

        self.best_idx = np.argmax(self.probs)
        self.best_prob = self.probs[self.best_idx]
        self.k = k

    def step(self, k):
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0
