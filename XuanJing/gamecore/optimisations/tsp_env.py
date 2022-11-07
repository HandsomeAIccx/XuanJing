import numpy as np


class TspEnv(object):
    def __init__(self, city_num, batch_size):
        self.city_num = city_num
        self.batch_size = batch_size

    def reset(self):
        """
        state: 城市的数量与城市的横纵坐标。
        """
        state = np.random.rand(self.batch_size, self.city_num, 2)
        return np.array(state, dtype=np.float32)

    def step(self, actions):
        rewards = None
        return rewards
