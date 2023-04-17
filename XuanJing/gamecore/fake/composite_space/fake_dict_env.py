import random
import numpy as np
from gymnasium.spaces import Dict, Box, Discrete


class FakeDictEnv(object):
    def __init__(self) -> None:
        self.observation_space = Dict({"position": Box(-1, 1, shape=(2,)), "color": Discrete(3)})
        self.action_space = Dict({"position": Box(-1, 1, shape=(2,)), "color": Discrete(3)})

    def reset(self):
        return self.observation_space.sample()

    def step(self, action):
        assert action in self.action_space, f"action {action} is not in action_space {self.action_space}"
        next_obs = self.observation_space.sample()
        reward, done, info = 1.0, random.choice([True, False]), {}
        return next_obs, reward, done, info


if __name__ == "__main__":
    # 1. FakeDictEnv
    fakeDictEnv = FakeDictEnv()
    obs = fakeDictEnv.reset()
    done = False
    while not done:
        action = fakeDictEnv.action_space.sample()
        next_obs, reward, done, info = fakeDictEnv.step(action)
        obs = next_obs
        print(obs)
    print(done)
