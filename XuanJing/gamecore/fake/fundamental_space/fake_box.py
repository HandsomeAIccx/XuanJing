import random
import numpy as np
from gymnasium.spaces import Box


class FakeIdenticalBoxEnv(object):
    def __init__(self) -> None:
        self.observation_space = Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        self.action_space = Box(low=-1.0, high=2.0, shape=(1, 4), dtype=np.float32)

    def reset(self):
        return self.observation_space.sample()

    def step(self, action):
        assert action in self.action_space, f"action {action} is not in action_space {self.action_space}"
        next_obs = self.observation_space.sample()
        reward, done, info = 1.0, random.choice([True, False]), {}
        return next_obs, reward, done, info


class FakeIndependBoxEnv(object):
    def __init__(self) -> None:
        self.observation_space = Box(low=np.array([-1.0, -2.0]), high=np.array([2.0, 4.0]), dtype=np.float32)
        self.action_space = Box(low=np.array([-1.0, -2.0]), high=np.array([2.0, 4.0]), dtype=np.float32)

    def reset(self):
        obs = self.observation_space.sample()
        # if obs.ndim == 1:
        #     obs = np.expand_dims(obs, axis=0)
        return obs

    def step(self, action):
        assert action in self.action_space, f"action {action} is not in action_space {self.action_space}"
        next_obs = self.observation_space.sample()
        # if next_obs.ndim == 1:
        #     next_obs = np.expand_dims(next_obs, axis=0)
        reward, done, info = 1.0, random.choice([True, False]), {}
        return next_obs, reward, done, info


if __name__ == "__main__":
    # 1. FakeIdenticalBoxEnv
    fakeIdenticalBoxEnv = FakeIdenticalBoxEnv()
    obs = fakeIdenticalBoxEnv.reset()
    done = False
    while not done:
        action = fakeIdenticalBoxEnv.action_space.sample()
        next_obs, reward, done, info = fakeIdenticalBoxEnv.step(action)
        obs = next_obs
        print(obs.shape)
    print(done)

    # 2. FakeIndependBoxEnv
    fakeIndependBoxEnv = FakeIndependBoxEnv()
    obs = fakeIndependBoxEnv.reset()
    done = False
    while not done:
        action = fakeIndependBoxEnv.action_space.sample()
        next_obs, reward, done, info = fakeIndependBoxEnv.step(action)
        obs = next_obs
        print(obs.shape)
    print(done)
