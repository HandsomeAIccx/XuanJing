import random
from gymnasium.spaces import Discrete


class FakeDiscreteEnv(object):
    def __init__(self) -> None:
        self.observation_space = Discrete(3, start=-1, seed=42)  # {-1, 0, 1}
        self.action_space = Discrete(2, seed=42)  # {0, 1}

    def reset(self):
        return self.observation_space.sample()

    def step(self, action):
        assert action in self.action_space, f"action {action} is not in action_space {self.action_space}"
        next_obs = self.observation_space.sample()
        reward, done, info = 1.0, random.choice([True, False]), {}
        return next_obs, reward, done, info


if __name__ == "__main__":
    fakeDiscreteEnv = FakeDiscreteEnv()
    obs = fakeDiscreteEnv.reset()
    done = False
    while not done:
        action = fakeDiscreteEnv.action_space.sample()
        next_obs, reward, done, info = fakeDiscreteEnv.step(action)
        obs = next_obs
        print(obs)
    print(done)
