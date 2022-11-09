# -*- coding: utf-8 -*-
# @Time    : 2022/11/8 9:25 下午
# @Author  : Zhiqiang He
# @Email   : tinyzqh@163.com
# @File    : dealer.py
# @Software: PyCharm

import enum
import random

BackJackGameCard2EnvCard = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "T": 10,
    "J": 10,
    "Q": 10,
    "K": 10,
    "A": 1
}


class Suit(enum.Enum):
    spades = "spades"
    clubs = "clubs"
    diamonds = "diamonds"
    hearts = "hearts"


class Card(object):
    def __init__(self, suit, rank_str, rank_value):
        self.suit = suit
        self.rank_str = rank_str
        self.rank_value = rank_value

    def __str__(self):
        return self.rank_str + " of " + self.suit.value


class BlackjackDealer(object):
    def __init__(self, deck_num=1):
        self.cards = []
        for i in range(deck_num):
            for suit in Suit:
                for rank, value in BackJackGameCard2EnvCard.items():
                    self.cards.append(Card(suit, rank, value))

    def shuffle(self):
        random.shuffle(self.cards)

    def deal(self):
        return self.cards.pop(0)

    def __len__(self):
        return len(self.cards)


if __name__ == "__main__":
    deck = BlackjackDealer()
    print(deck.cards)


