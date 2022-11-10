import gym
from gym.spaces import Discrete, Tuple
from gym import spaces


class KuhnPoker(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(2)  # pass or bet.
        self.done = False


    def reset(self):
        self.done = False
        self.current_player = 0
        self.history = []
        return None, None

    def step(self, action):
        assert self.action_space.contains(action), f"{action} is not contain in action space!"


if __name__ == "__main__":
    pass