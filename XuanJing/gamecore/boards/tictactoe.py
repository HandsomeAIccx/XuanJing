import random

import gym
import numpy as np
from gym import spaces
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    List,
    Optional,
    SupportsFloat,
    Tuple,
    TypeVar,
    Union,
)

ObsType = TypeVar("ObsType")


class TicTacToe(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Discrete(9)

        self.board = None
        self.pos2strMap = {
            -1: "   ",
            0: " O ",
            1: " X "
        }

        self.player1 = 0
        self.player2 = 1

        self.current_player = self.player1

    def reset(
            self,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[dict] = None,
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
        self.board = [[-1 for _ in range(3)] for _ in range(3)]

    def step(self, action):
        self.board[int(action / 3)][action % 3] = self.current_player
        self.current_player = self.player1 if self.current_player == self.player2 else self.player2
        print(self.board)
        self.render()

    def sample(self):
        return random.choice(self._legal_action())

    def render(self, mode="human"):
        print("-----------")
        for row in self.board:
            print("{}|{}|{}".format(self.pos2strMap[row[0]],
                                       self.pos2strMap[row[1]],
                                       self.pos2strMap[row[2]]))
            print("-----------")

    def _legal_action(self):
        """Get Legal Action！"""
        legal_action = [i for i in range(self.action_space.n) if self.board[int(i / 3)][i % 3] == -1]
        return legal_action


if __name__ == "__main__":
    env = TicTacToe()
    env.reset()
    env.render()
    for i in range(5):
        action = env.sample()
        env.step(action)
    print("a")

