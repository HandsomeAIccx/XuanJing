import random
from gymnasium.spaces import Tuple, Box, Discrete


class FakeTupleEnv(object):
    def __init__(self) -> None:
        self.observation_space = Tuple((Discrete(2), Box(-1, 1, shape=(2,))))
        self.action_space = Tuple((Discrete(2), Box(-1, 1, shape=(2,))))

    def reset(self):
        return self.observation_space.sample()

    def step(self, action):
        assert action in self.action_space, f"action {action} is not in action_space {self.action_space}"
        next_obs = self.observation_space.sample()
        reward, done, info = 1.0, random.choice([True, False]), {}
        return next_obs, reward, done, info


if __name__ == "__main__":
    # 1. FakeTupleEnv
    fakeTupleEnv = FakeTupleEnv()
    obs = fakeTupleEnv.reset()
    done = False
    while not done:
        action = fakeTupleEnv.action_space.sample()
        next_obs, reward, done, info = fakeTupleEnv.step(action)
        obs = next_obs
        print(obs)
    print(done)
