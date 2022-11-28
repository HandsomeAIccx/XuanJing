import numpy as np


class SolverAgent(object):
    def __init__(self, env):
        self.env = env
        self.counts = np.zeros(self.env.k)
        self.regret = 0.0
        self.action_history = []
        self.regret_history = []

    def take_action(self):
        pass

    def update_regret(self, k):
        self.regret = self.env.best_prob - self.env.probs[k]
        self.regret_history.append(self.regret)

