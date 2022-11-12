import gym
import random
from gym import spaces
from copy import deepcopy


class KuhnPoker(gym.Env):
    def __init__(self):
        self.cards = [1, 2, 3]
        random.shuffle(self.cards)
        self.action_space = spaces.Discrete(2)  # check or bet.

        self.players = ["player1", "player2"]

        self.player_hand = None
        self.player_history = None
        self.history = None

        self.player_reward = None

        self.current_player = None
        self.opponent_player = None
        self.info_sets = None
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
        self.history = []
        self.player_reward = {
            "player1": 0,
            "player2": 0
        }
        self.info_sets = {
            "player1": InfoSet("player1"),
            "player2": InfoSet("player2")
        }

        return self.done, self.player_reward, self._get_infoset()

    def step(self, action):
        assert self.action_space.contains(action), f"{action} is not contain in action space!"
        assert self.current_player in ["player1", "player2"], "player not in [player1, player2]!"

        self.player_history[self.current_player].append(action)
        self.history.append(action)

        if self._is_terminal():  # terminal
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
        return self.done, self.player_reward, self._get_infoset()

    def _is_terminal(self):
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

    def _get_infoset(self):
        self.info_sets[self.current_player].player_identity = self.current_player
        self.info_sets[self.current_player].player_hand_card = self.player_hand[self.current_player]
        self.info_sets[self.current_player].player_history = self.player_history[self.current_player]
        self.info_sets[self.current_player].opponent_history = self.player_history[self.opponent_player]
        self.info_sets[self.current_player].history = self.history
        self.info_sets[self.current_player].legal_action = [0, 1]
        return deepcopy(self.info_sets[self.current_player])

    def render(self, mode="human"):
        print(self._get_infoset())


class InfoSet(object):
    def __init__(self, player_identity):
        self.player_identity = player_identity
        self.player_hand_card = None
        self.player_history = None
        self.opponent_history = None
        self.history = None
        self.legal_action = None

    def __str__(self):
        return "identity:{} | hand_card:{} | play_history:{} | opp_history:{} | history:{} | legal_action:{}".format(
            self.player_identity,
            self.player_hand_card,
            self.player_history,
            self.opponent_history,
            self.history,
            self.legal_action)

