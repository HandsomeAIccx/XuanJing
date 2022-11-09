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

from XuanJing.gamecore.cards.blackjack.dealer import BlackjackDealer


def usable_ace(hands):  # Does this hand have a usable ace?
    hand_num = [hand.rank_value for hand in hands]
    return 1 in hand_num and sum(hand_num) + 10 <= 21


def sum_hand(hands):  # Return current hand total
    hand_num = [hand.rank_value for hand in hands]
    if 1 in hand_num and sum(hand_num) + 10 <= 21:  # exit Ace, and use Ace as 11 < 21.
        return sum(hand_num) + 10
    return sum(hand_num)


class BlackJackEnv(gym.Env):
    def __init__(self):
        super(BlackJackEnv, self).__init__()
        self.deck_num = 1
        self.bj_dealer = BlackjackDealer(deck_num=self.deck_num)

        self.player_hand = []
        self.dealer_hand = []
        self.dealer_up_card = None

        # hit = 0, stand = 1
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(32), spaces.Discrete(11), spaces.Discrete(2))
        )
        self.done = False

    def reset(
            self,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[dict] = None,
    ) -> Union[ObsType, Tuple[ObsType, dict]]:

        self.bj_dealer.cards += self.player_hand + self.dealer_hand
        assert len(self.bj_dealer.cards) == self.deck_num * 52, "Incomplete Recovery Deck Number！"
        self.bj_dealer.shuffle()

        self.done = False

        self.player_hand = [self.bj_dealer.deal(), self.bj_dealer.deal()]
        self.dealer_hand = [self.bj_dealer.deal(), self.bj_dealer.deal()]
        self.dealer_up_card = self.dealer_hand[0]
        return self._get_obs(), {}

    def step(self, action):
        assert self.action_space.contains(action), f"{action} is not contain in action space!"

        if action:  # hit: add card to players hand and return.
            self.player_hand.append(self.bj_dealer.deal())
            if sum_hand(self.player_hand) > 21:
                terminated = True
                reward = -1.0
            else:
                terminated = False
                reward = 0.0
        else:  # stick: play out the dealers hand, and score
            terminated = True
            while sum_hand(self.dealer_hand) < 17:  # less 17 point, the dealer hit.
                self.dealer_hand.append(self.bj_dealer.deal())

            player_score = 0 if sum_hand(self.player_hand) > 21 else sum_hand(self.player_hand)
            dealer_score = 0 if sum_hand(self.dealer_hand) > 21 else sum_hand(self.dealer_hand)
            reward = float(player_score > dealer_score) - float(player_score < dealer_score)

        return self._get_obs(), reward, terminated, False, {}

    def _get_obs(self):
        return sum_hand(self.player_hand), self.dealer_hand[0], usable_ace(self.player_hand)

    def close(self):
        pass
