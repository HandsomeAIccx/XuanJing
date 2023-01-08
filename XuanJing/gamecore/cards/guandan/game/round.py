# -*- coding: utf-8 -*-
""" Implement Guandan Round class
"""

import functools
import numpy as np
import random
import sys

from XuanJing.gamecore.cards.guandan.game.utils import cards2str
from XuanJing.gamecore.cards.guandan.game.utils import CARD_RANK, CARD_RANK_STR, CARD_RANK_STR_INDEX
from XuanJing.gamecore.cards.guandan.game.utils import Card
from XuanJing.gamecore.cards.guandan.game.dealer import GuandanDealer as Dealer

sys.setrecursionlimit(15000)


class GuandanRound:
    """
    Round can call other Classes' functions to keep the game running
    """

    def __init__(self, officer, players, win_group, win_history, winners):
        """
        Args:
            officer: 两队的参谋.
            players: 两队的玩家实例化对象
            win_group:
            win_history: 记录历史输赢情况.
        """
        self.np_random = np.random.RandomState()
        # self.played_cards = played_cards
        self.officer = officer
        self.trace = []

        self.greater_player = None
        self.dealer = Dealer(self.np_random, self)
        # self.deck_str = cards2str(self.dealer.deck)
        self.seen_cards = ""
        self.winner_group = win_group
        self.winners = winners
        self.tribute_cards = None  # 进贡牌
        self.tribute_players = None  # 进贡玩家
        self.detribute = False
        self.players = players

        self.dealer.init(self.players)

        if not self.winner_group:  # 如果是游戏的第一小局，随机选择一个玩家开始游戏
            self.current_player = random.randint(0, 3)
        else:  # 否则进贡
            # TODO 还贡
            self.current_player, self.detribute = self.pay_tribute()

        self.public = {
            # 'deck': self.deck_str,
            'seen_cards': self.seen_cards,
            # 'winner_group': self.winner_group,
            'trace': self.trace
        }

    def proceed_round(self, player, action):
        """
        Call other Classes's functions to keep one round running.
        进行一轮
        Args:
            player (object): object of Player
            action (str): string of legal specific action
        Returns:
            object of Player: player who played current biggest cards.
        """

        self.greater_player = player.play(action, self.greater_player)  # 出牌
        return self.greater_player

    def find_last_greater_player_id_in_trace(self):
        """
        Find the last greater_player's id in trace.
        找到出牌最大的玩家
        Returns:
            The last greater_player's id in trace
        """
        for i in range(len(self.trace) - 1, -1, -1):
            _id, action = self.trace[i]
            # 找到最后一个出牌的玩家
            if action != 'pass':
                return _id
        return None

    def find_last_played_cards_in_trace(self, player_id):
        """
        找到玩家上一轮出的牌
        Find the player_id's last played_cards in trace
        Returns:
            The player_id's last played_cards in trace
        """
        for i in range(len(self.trace) - 1, -1, -1):
            _id, action = self.trace[i]
            if _id == player_id and action != 'pass':
                return action
        return None

    def sort_card(self, card_1, card_2):
        """
        按照牌的大小排序
        Compare the rank of two cards of Card object

        Args:
            card_1 (object): object of Card
            card_2 (object): object of card
            :param current_officer: 当前的参谋
        """
        key = []
        for card in [card_1, card_2]:
            # print(card)
            if card.rank == '':
                key.append(CARD_RANK.index(card.suit))
            else:
                key.append(CARD_RANK.index(card.rank))

        # #  如果card_1是参谋且card_2不是参谋和大小鬼
        # if card_1.rank == self.officer and card_2.rank != self.officer and key[1] < 13:
        #     return 1
        # # 如果card_2是参谋且card_1不是参谋和大小鬼
        # if card_2.rank == self.officer and card_1.rank != self.officer and key[0] < 13:
        #     return -1

        if key[0] > key[1]:
            return 1
        if key[0] < key[1]:
            return -1
        return 0

    def pay_tribute(self):
        """
        # 进贡
        :return: 下一步进发出动作的玩家，是否为还贡
        """
        # 单贡
        if self.winners >= 3:
            # 进贡的玩家
            tribute_player = None
            # 还牌的玩家
            detribute_player = self.winners[0]
            for player in self.players:
                if player.player_id not in self.winners:
                    tribute_player = player.player_id
                    break
            # 如果抓到2个大王，则抗贡
            if tribute_player.count_RJ() == 2:
                self.seen_cards = cards2str([Card("RJ", ""), Card("RJ", "")])
                # 上游出牌
                return self.winners[0], False
            else:

                self.tribute_cards = [tribute_player.get_tribute_card(self.officer)]
                seen_cards = self.tribute_cards
                seen_cards.sort(key=functools.cmp_to_key(self.sort_card))
                self.seen_cards = cards2str(seen_cards)
                self.tribute_players = [tribute_player]
                # 上游玩家还贡
                return self.winners[0], True
        # 双上
        elif self.winners == 2:
            tribute_players = []
            for player in self.players:
                if player.player_id not in self.winners:
                    tribute_players.append(player)
            # 双下抓到两张大王，抗贡
            if tribute_players[0].count_RJ() + tribute_players[1].count_RJ >= 2:
                return self.winners[0], False
            else:
                self.tribute_cards.append(tribute_players[0].get_tribute_card())
                self.tribute_cards.append(tribute_players[1].get_tribute_card())
                seen_cards = self.tribute_cards
                seen_cards.sort(key=functools.cmp_to_key(self.sort_card))
                self.seen_cards = cards2str(seen_cards)
                self.tribute_players = tribute_players
                # 还贡
                return self.winners[0], True
