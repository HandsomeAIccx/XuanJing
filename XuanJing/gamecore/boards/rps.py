import random
import gym
from gym import spaces
from copy import deepcopy


class RockPaperScissors(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(3)  # R, P, S

        self.player_space = ["player1", "player2"]

        self.action_utility = [
            [0, -1, 2],
            [1, 0, -2],
            [-2, 2, 0]
        ]  # each row is going to be the user, col going to be the opponent.

        self.players = None
        self.player_history = None
        self.player_reward = None

        self.history = None

        self.current_player = None
        self.opponent_player = None

        self.info_sets = None
        self.done = False

    def reset(self):
        self.players = ["player1", "player2"]
        random.shuffle(self.players)
        self.current_player = self.players.pop()
        self.opponent_player = self.players.pop()

        self.info_sets = {
            "player1": InfoSet("player1"),
            "player2": InfoSet("player2")
        }

        self.player_history = {
            "player1": [],
            "player2": []
        }
        self.history = []

        self.player_reward = {
            "player1": 0,
            "player2": 0
        }

        self.done = False
        return self._get_infoset(), self.done, self.player_reward

    def render(self, mode="human"):
        print(self._get_infoset())

    def step(self, action):
        assert self.action_space.contains(action), f"{action} is not contain in action space!"

        self.player_history[self.current_player].append(action)
        self.history.append(action)
        if self._is_terminal():
            self.done = True
            player_action = self.player_history[self.current_player][0]
            opp_action = self.player_history[self.opponent_player][0]
            self.player_reward[self.current_player] = self.action_utility[player_action][opp_action]
            self.player_reward[self.opponent_player] = self.action_utility[opp_action][player_action]
        else:
            pass
        self.current_player, self.opponent_player = self.opponent_player, self.current_player
        return self._get_infoset(), self.done, self.player_reward

    def _get_infoset(self):
        self.info_sets[self.current_player].player_identity = self.current_player
        return deepcopy(self.info_sets[self.current_player])

    def _is_terminal(self):
        if len(self.history) == 2: return True
        else: return False


class InfoSet(object):
    def __init__(self, player_identity):
        self.player_identity = player_identity

    def __str__(self):
        return "identity: {}".format(self.player_identity)


if __name__ == "__main__":
    env = RockPaperScissors()
    env.reset()
    for i in range(100):
        env.render()
        action = int(input())
        obs, done, reward = env.step(action)
        if done:
            print("Game Done! Reward is : ", reward)
            env.reset()
            print("Start New Game!")
