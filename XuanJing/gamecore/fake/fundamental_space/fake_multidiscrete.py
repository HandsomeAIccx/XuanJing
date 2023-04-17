import random
import numpy as np

from gymnasium.spaces import MultiDiscrete


class FakeMultiDiscreteEnv(object):
    def __init__(self) -> None:
        self.observation_space = MultiDiscrete(np.array([[1, 2], [3, 4]]))
        self.action_space = MultiDiscrete(np.array([[1, 2, 3, 4]]))

    def reset(self):
        return self.observation_space.sample()

    def step(self, action):
        assert action in self.action_space, f"action {action} is not in action_space {self.action_space}"
        next_obs = self.observation_space.sample()
        reward, done, info = 1.0, random.choice([True, False]), {}
        return next_obs, reward, done, info


if __name__ == "__main__":
    fakeMultiDiscreteEnv = FakeMultiDiscreteEnv()
    obs = fakeMultiDiscreteEnv.reset()
    done = False
    while not done:
        action = fakeMultiDiscreteEnv.action_space.sample()
        next_obs, reward, done, info = fakeMultiDiscreteEnv.step(action)
        obs = next_obs
        print(obs)
    print(done)
