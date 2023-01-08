import functools
from heapq import merge
import numpy as np
import copy

from XuanJing.gamecore.cards.guandan.game.utils import cards2str, CARD_RANK_STR, cards2str_with_suit
from XuanJing.gamecore.cards.guandan.game.player import GuandanPlayer as Player
from XuanJing.gamecore.cards.guandan.game.round import GuandanRound as Round
from XuanJing.gamecore.cards.guandan.game.judger import GuandanJudger as Judger


class GuanDanInfoSet(object):
    def __init__(self):
        """
        The InfoSet for GuanDan.
        """
        self.player_id = None
        self.num_cards_left = None
        self.player_hand_cards = None
        self.history = None
        self.other_hand_cards = None
        self.legal_action = None
        self.seen_cards = None
        self.trace = None
        self.perfect = None


class GuandanGame:
    """
    Provide game APIs for env to run guandan and get corresponding state
    information.
    """

    def __init__(self):
        # self.np_random = np.random.RandomState()
        self.num_players = 4
        self.winner_id = []

        self.group_officer = ['2', '2']  # 当前两组玩家的参谋
        self.win_history = []  # 记录每轮的输赢情况

    def reset(self):
        """
        Reset The round of Game.
        Returns:
            dict: first state in one game
            int: current player's id
        """
        # TODO hzq 判断大轮reset还是小轮reset。
        # initialize public variables
        self.winner_group = None
        self.winner_id = []
        self.history = []

        # initialize players
        self.players = [Player(num) for num in range(self.num_players)]

        # 出过的牌
        # CARD_RANK_STR为所有牌面值
        # self.played_cards = [np.zeros((len(CARD_RANK_STR),), dtype=np.int) for _ in range(self.num_players)]

        # 初始化局
        self.round = Round(self.group_officer, self.players, self.winner_group, self.win_history, self.winner_id)
        # self.round.initiate(self.players, self.winner_group, self.winner_id)

        # 初始化裁判
        self.judger = Judger(self.players, self.group_officer)

        # get state of first player
        player_id = self.round.current_player
        # print(player_id)
        self.state = self.get_state(player_id)

        return self.state

    def step(self, action):
        """
        Perform one draw of the game
        Args:
            action (str): specific action of doudizhu. Eg: '33344'
        Returns:
            dict: next player's state
            int: next player's id
        """

        # perfrom action
        player = self.players[self.round.current_player]
        self.round.proceed_round(player, action)
        # print(player.player_id, action)
        # print(cards2str(player.current_hand))

        # 如果出牌
        if action != 'pass':
            # 当前可以出的牌
            self.judger.calc_playable_cards(player)
        #
        if self.judger.judge_game(self.players, self.round.current_player):
            self.winner_id.append(self.round.current_player)
        # 轮到下一位玩家出牌
        next_id = (player.player_id + 1) % len(self.players)
        while len(self.players[next_id].current_hand) == 0:
            if next_id not in self.winner_id:
                self.winner_id.append(next_id)
            next_id = (next_id + 1) % len(self.players)
        self.round.current_player = next_id

        # 获得下一位玩家的状态
        state = self.get_state(next_id)
        self.state = state

        return state

    # 获取当前玩家的状态
    def get_state(self, player_id):
        """
        Return player's state
        Args:
            player_id (int): player id
        Returns:
            (dict): The state of the player
        """

        state_infoset = GuanDanInfoSet()

        # 当前玩家
        current_player = self.players[player_id]
        # 其他玩家
        other_players = [player for player in self.players if player.player_id != current_player.player_id]

        # 其它玩家手中的牌
        others_hands = []
        for p in other_players:
            others_hands = merge(others_hands, p.current_hand, key=functools.cmp_to_key(self.round.sort_card))

        # 如果当前小局结束
        if self.is_over():
            # 清空当前动作
            actions = []
        # 如果当前小局没有结束
        else:
            actions = list(
                current_player.available_actions(self.round.officer,
                                                 self.judger.count_heart_officer(current_player.group_id, current_player.current_hand),
                                                 self.round.greater_player, self.judger))

        # 当前玩家带花色的手牌
        player_hand_with_suit = self._get_current_player_hand_with_suit(current_player)

        state_infoset.player_id = current_player.player_id
        state_infoset.player_hand_cards = player_hand_with_suit
        state_infoset.other_hand_cards = cards2str_with_suit(others_hands)
        state_infoset.legal_action = actions

        # 当前所有玩家手上的牌的数量
        state_infoset.num_cards_left = {player.player_id: len(player.current_hand) for player in self.players}

        state_infoset.seen_cards = self.round.public["seen_cards"]
        state_infoset.trace = self.round.public["trace"]

        state_infoset.history = {player.player_id: player.all_history for player in self.players}
        state_infoset.perfect = {player.player_id: cards2str_with_suit(player.current_hand) for player in self.players}
        return copy.deepcopy(state_infoset)

    def is_over(self):
        """
        Judge whether a game is over
        Returns:
            Bool: True(over) / False(not over)
        """
        # 只有一位玩家出完牌
        if len(self.winner_id) < 2:
            return False
        elif len(self.winner_id) == 2:
            player1 = self.winner_id[0]
            player2 = self.winner_id[1]
            # 双上
            if (player1 % 2) == (player2 % 2):
                return True
            # 不是双上，要等第三位玩家出完牌
            else:
                return False
        # 前三位玩家都已经出完牌，当前小局结束
        elif len(self.winner_id) >= 3:
            return True
        return False

    # 获取其他玩家当前手上的牌
    def _get_others_current_hand(self, player):
        other_players = []
        for p in self.players:
            if p.player_id != player.player_id:
                other_players.append(p)

        others_hand = []
        for p in other_players:
            others_hand = merge(others_hand, p.current_hand, key=functools.cmp_to_key(self.round.sort_card))
        return cards2str(others_hand)

    def _get_current_player_hand_with_suit(self, player):
        player_hand = []
        player_hand = merge(player_hand, player.current_hand, key=functools.cmp_to_key(self.round.sort_card))
        return cards2str_with_suit(player_hand)
