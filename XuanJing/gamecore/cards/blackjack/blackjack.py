# -*- coding: utf-8 -*-
# @Time    : 2022/11/8 8:52 下午
# @Author  : Zhiqiang He
# @Email   : tinyzqh@163.com
# @File    : blackjack.py
# @Software: PyCharm

import gym
import numpy as np
from gym import spaces

from XuanJing.gamecore.cards.blackjack.dealer import BlackjackDealer


class BlackJackEnv(gym.Env):
    def __init__(self):
        super(BlackJackEnv, self).__init__()
        self.deck_num = 1
        self.bj_dealer = BlackjackDealer(deck_num=self.deck_num)

        self.player_hand = []
        self.dealer_hand = []
        self.dealer_up_card = None

        self.reward_options = {"lose": -100, "tie": 0, "win": 100}

        # hit = 0, stand = 1
        self.action_space = spaces.Discrete(2)

        self.observation_space = spaces.Tuple((spaces.Discrete(18), spaces.Discrete(10)))
        self.done = False

    def reset(self):
        self.bj_dealer.cards += self.player_hand + self.dealer_hand
        assert len(self.bj_dealer.cards) == self.deck_num * 54, "Incomplete Recovery Deck Number！"
        self.bj_dealer.shuffle()

        self.done = False

        self.player_hand = [self.bj_dealer.deal(), self.bj_dealer.deal()]
        self.dealer_hand = [self.bj_dealer.deal(), self.bj_dealer.deal()]
        self.dealer_up_card = self.dealer_hand[0]

        # calculate the value of the agent's hand.
        self.player_value = self._compute_bj_card_value(self.player_hand)

        # This makes the possible range of 2 through 20 into 1 through 18
        player_value_obs = self.player_value - 2

        # Subtract by 1 to fit the possible observation range of 1 to 10.
        upcard_value_obs = self._compute_bj_card_value([self.dealer_upcard]) - 1
        obs = np.array([player_value_obs, upcard_value_obs])
        return obs

    def _take_action(self, action):
        if action == 0:  # hit
            self.player_hand.append(self.bj_dealer.deal())

        self.player_value = self._compute_bj_card_value(self.player_hand)

    def step(self, action):
        self._take_action()

        self.done = action == 1 or self.player_value >= 21

        rewards = 0
        if self.done:
            if self.player_value > 21:  # above 21, player loses automatically.
                rewards = self.reward_options["lose"]
            elif self.player_value == 21:  # blackjack! Player wins automatically.
                rewards = self.reward_options["win"]
            else:
                dealer_value, self.dealer_hand, self.bj_deck = self.dealer_turn(self.dealer_hand, self.bj_deck)

                if dealer_value > 21:  # dealer above 21, player wins automatically
                    rewards = self.reward_options["win"]
                elif dealer_value == 21:  # dealer has blackjack, player loses automatically
                    rewards = self.reward_options["lose"]
                else:  # dealer and player have values less than 21.
                    if self.player_value > dealer_value:  # player closer to 21, player wins.
                        rewards = self.reward_options["win"]
                    elif self.player_value < dealer_value:  # dealer closer to 21, dealer wins.
                        rewards = self.reward_options["lose"]
                    else:
                        rewards = self.reward_options["tie"]
        player_value_obs = self.player_value - 2
        upcard_value_obs = self._compute_bj_card_value([self.dealer_up_card]) - 1
        obs = np.array([player_value_obs, upcard_value_obs])
        return obs, rewards, self.done, {}

    @staticmethod
    def dealer_turn(dealer_hand, deck):
        # Calculate dealer hand's value.
        dealer_value = BlackJackEnv._compute_bj_card_value(dealer_hand)

        # Define dealer policy (is fixed to official rules)

        # The dealer keeps hitting until their total is 17 or more
        while dealer_value < 17:
            # hit
            dealer_hand.append(deck.deal())
            dealer_value = BlackJackEnv._compute_bj_card_value(dealer_hand)

        return dealer_value, dealer_hand, deck

    @staticmethod
    def _compute_bj_card_value(hand_card):
        """define logic for evaluating the value of the hand card."""
        num_ace = 0
        use_one = 0
        for card in hand_card:
            if card.rank_str == "ace":
                num_ace += 1
                use_one += card.rank_value[0]
            else:
                use_one += card.rank_value

        if num_ace > 0:
            ace_counter = 0
            while ace_counter < num_ace:
                use_eleven = use_one + 10
                if use_eleven > 21:
                    return use_one
                elif use_eleven >= 18 and use_eleven <= 21:
                    return use_eleven
                else:
                    use_one = use_eleven
                ace_counter += 1
            return use_one
        else:
            return use_one


if __name__ == "__main__":
    bj_env = BlackJackEnv()
    print("a")