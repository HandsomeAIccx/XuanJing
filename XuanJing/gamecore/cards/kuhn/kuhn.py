import random

import gym
from gym import spaces


class Agent(object):
    def __init__(self):
        pass

    def act(self, info_set):
        pass


class KuhnPoker(gym.Env):
    def __init__(self):
        self.cards = [1, 2, 3]
        random.shuffle(self.cards)
        self.action_space = spaces.Discrete(2)  # check or bet.

        self.players = ["player1", "player2"]

        self.player_hand = {
            "player1": self.cards.pop(),
            "player2": self.cards.pop()
        }
        self.player_history = {
            "player1": [],
            "player2": []
        }
        self.player_reward = {
            "player1": 0,
            "player2": 0
        }

        self.current_player = None
        self.opponent_player = None

        self.history = []

        self.done = False

    def reset(self):
        self.done = False

        self.cards = [1, 2, 3]
        random.shuffle(self.cards)

        self.players = ["player1", "player2"]
        random.shuffle(self.players)
        self.current_player = self.players.pop()
        self.opponent_player = self.players.pop()

        self.player_hand = {
            "player1": self.cards.pop(),
            "player2": self.cards.pop()
        }
        self.player_history = {
            "player1": [],
            "player2": []
        }
        self.player_reward = {
            "player1": 0,
            "player2": 0
        }

        return self.done, self.player_reward, self.player_hand

    def step(self, action):
        assert self.action_space.contains(action), f"{action} is not contain in action space!"
        assert self.current_player in ["player1", "player2"], "player not in [player1, player2]!"

        self.player_history[self.current_player].append(action)
        self.history.append(action)

        if self.is_terminal():  # terminal
            self.done = True
            assert len(self.history) >= 2, "len of history < 2!"
            if self.history[-2:] == [1, 0]:  # bit , pass
                self.player_reward[self.current_player] = -1
                self.player_reward[self.opponent_player] = 1
            elif self.history[-2:] == [1, 1]:  # bit, bit
                if self.player_hand[self.current_player] > self.player_hand[self.opponent_player]:
                    self.player_reward[self.current_player] = 2
                    self.player_reward[self.opponent_player] = -2
                elif self.player_hand[self.current_player] < self.player_hand[self.opponent_player]:
                    self.player_reward[self.current_player] = -2
                    self.player_reward[self.opponent_player] = 2
                else:
                    raise ValueError("No Equal!")
            elif self.history[-2:] == [0, 0]:  # pass, pass
                if self.player_hand[self.current_player] > self.player_hand[self.opponent_player]:
                    self.player_reward[self.current_player] = 1
                    self.player_reward[self.opponent_player] = -1
                elif self.player_hand[self.current_player] < self.player_hand[self.opponent_player]:
                    self.player_reward[self.current_player] = -1
                    self.player_reward[self.opponent_player] = 1
                else:
                    raise ValueError("No Equal!")
        else:
            pass

        # change player!
        self.current_player, self.opponent_player = self.opponent_player, self.current_player
        return self.done, self.player_reward, self.player_hand

    def is_terminal(self):
        if len(self.history) == 3:
            return True
        elif len(self.history) == 2:
            if self.history[0] == 0 and self.history[1] == 0:
                return True
            elif self.history[0] == 1 and self.history[1] == 1:
                return True
            elif self.history[0] == 1 and self.history[1] == 0:
                return True
        return False


if __name__ == "__main__":
    env = KuhnPoker()
    env.reset()
    for i in range(10):
        action = env.action_space.sample()
        done, reward, player_hand = env.step(action)
        if done:
            env.reset()
        print("a")