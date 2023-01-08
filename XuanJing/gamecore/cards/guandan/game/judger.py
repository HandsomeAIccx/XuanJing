import copy

import numpy as np
import collections
import itertools


from XuanJing.gamecore.cards.guandan.game.utils import CARD_RANK_STR, CARD_RANK_STR_INDEX, INDEX_CARD
from XuanJing.gamecore.cards.guandan.game.utils import cards2str, contains_cards


class GuandanJudger:
    """ Determine what cards a player can play
    """

    def __init__(self, players, officer):
        """
        Initilize the Judger class
        """
        # 4位玩家
        # 当前可以出的牌
        self.playable_cards = [set() for _ in range(4)]
        # 已经出过的牌
        self._recorded_removed_playable_cards = [[] for _ in range(4)]
        self.officer = officer
        for player in players:
            player_id = player.player_id
            player_group_id = player.group_id
            # 玩家当前手上的牌
            # current_hand = cards2str(player.current_hand)
            # 当前可以出的牌型
            self.playable_cards[player_id] = self.playable_cards_from_hand(player_group_id, player.current_hand)

    def get_afetr_serial_cards(self, legal_index, h_office_num, serial_length, repeat_num):
        """
        给定合法牌组index, 连续长度, 红心参谋数量, 返回带红心参谋的真实结果和红心参谋换牌之后的虚假结果.
        Tips: 返回顺延之后的连续结果。红心参谋牌起始情况并未考虑。需要在上一层逻辑中处理。
        Args:
            legal_index:
            office_num: 红心参谋的数量
        """

        assert [13] not in legal_index, "black jack in legal_index!"
        assert [14] not in legal_index, "red jack in legal_index!"
        assert h_office_num == 0

        chains_real = []
        chains_fake = []

        for left_index in range(len(legal_index)):
            cnt = 0
            chain_real = []
            chain_fake = []
            h_officer_num_left = h_office_num
            left_value = legal_index[left_index][0]
            if left_value == 12:
                right_indexes = [left_index] + list(range(left_index))
                ace_change = True  # Ace's value = 1
            else:
                right_indexes = list(range(left_index, len(legal_index)))
                ace_change = False  # Ace's value = 12
            while len(right_indexes) > 0:
                right_index = right_indexes.pop(0)
                right_value = legal_index[right_index][0]
                if cnt == 0:
                    for i in range(repeat_num):
                        chain_real.append(right_value)
                        chain_fake.append(right_value)
                    cnt += 1
                else:
                    assert cnt < serial_length, f"cnt more than {serial_length}."

                    last_right_value = -1 if ace_change and legal_index[right_index - 1][0] == 12 \
                        else legal_index[right_index - 1][0]  # 判断是否需要替换Ace's之后的计算值

                    if right_value - last_right_value == 1 or right_value - chain_fake[-1] == 1:
                        for i in range(repeat_num):
                            chain_real.append(right_value)
                            chain_fake.append(right_value)
                        cnt += 1
                    else:
                        if h_officer_num_left > 0:
                            chain_real.append(-1)
                            chain_fake.append(last_right_value + 1)
                            h_officer_num_left -= 1
                            cnt += 1
                            right_indexes.insert(0, right_index)   # 由于使用了红心参谋, 所以当前数字并未循环便利过，添加之后再进入循环
                        else:
                            break
                    if cnt == serial_length:
                        chains_real.append(chain_real)
                        chains_fake.append(chain_fake)
                        break

        return chains_real, chains_fake

    def serial_chain_indexes(self, card_dict, h_officer_num, office_str, repeat_num, serial_len):
        all_index_list = [[idx] for idx in list(range(len(CARD_RANK_STR_INDEX)))]
        all_index_list = [idx for idx in all_index_list if idx[0] <= 12]  # 移除掉RJ, BJ (大小王)
        # # 测试输入
        card_dict = {
            '2': 2,
            '3': 4,
            '4': 2,
            '5': 2,
            '6': 2,
            '7': 3,
            '8': 1,
            '9': 3,
            'T': 1,
            'J': 1,
            'Q': 1,
            'K': 0,
            'A': 2,
            'B': 1,
            'R': 1
        }
        office_str = '2'
        h_officer_num = 2
        repeat_num = 2
        serial_len = 3

        no_h_office_card_dict = copy.deepcopy(card_dict)  # 去除掉红心参谋之后的字典
        no_h_office_card_dict[office_str] -= h_officer_num
        no_h_office_card_count = np.array([no_h_office_card_dict[k] for k in CARD_RANK_STR])  # 去除掉红心参谋之后的count
        no_h_office_card_index_list = np.argwhere(no_h_office_card_count > repeat_num - 1)  # 去除掉红心参谋之后的index
        no_h_no_rjbj_indexes_list = [index for index in no_h_office_card_index_list if index[0] <= 12]  # 移除掉RJ, BJ (大小王)

        chains_real, chains_fake = self.get_afetr_serial_cards(no_h_no_rjbj_indexes_list, 0, serial_len, repeat_num)  # 先假设没有红心参谋

        if h_officer_num >= 1:  # 如果红心参谋数量大于等于1张, 并且使用1张红心参谋的情况
            missing_idx_list = [idx for idx in all_index_list if [idx] not in no_h_office_card_index_list]
            for miss_idx in missing_idx_list:
                copy_card_dict = copy.deepcopy(card_dict)
                copy_card_dict[INDEX_CARD[miss_idx[0]]] += 1   # 红心参谋填补一张空缺牌之后再重新计算.
                copy_count = np.array([copy_card_dict[INDEX_CARD[k[0]]] for k in all_index_list])
                copy_index = np.argwhere(copy_count > repeat_num - 1)
                chain_real_1, chains_fake_1 = self.get_afetr_serial_cards(copy_index, 0, serial_len, repeat_num)
                for real, fake in zip(chain_real_1, chains_fake_1):
                    if miss_idx[0] in real:
                        real[real.index(miss_idx[0])] = -1
                        chains_real.append(real)
                        chains_fake.append(fake)
        if h_officer_num == 2:  # 红心参谋数量为2张, 且两张全部使用的情况
            missing_idx_list = [idx for idx in all_index_list if [idx] not in no_h_office_card_index_list]
            missing_idx_comb = list(itertools.combinations(missing_idx_list, 2))
            for miss_idx in missing_idx_comb:
                copy_card_dict = copy.deepcopy(card_dict)
                copy_card_dict[INDEX_CARD[miss_idx[0][0]]] += 1  # 红心参谋填补两张空缺牌之后再重新计算.
                copy_card_dict[INDEX_CARD[miss_idx[1][0]]] += 1
                copy_count = np.array([copy_card_dict[INDEX_CARD[k[0]]] for k in all_index_list])
                copy_index = np.argwhere(copy_count > repeat_num - 1)
                chain_real_2, chains_fake_2 = self.get_afetr_serial_cards(copy_index, 0, serial_len, repeat_num)
                for real, fake in zip(chain_real_2, chains_fake_2):
                    if miss_idx[0][0] in real and miss_idx[1][0] in real:
                        if miss_idx[0][0] == miss_idx[1][0]:
                            real[real.index(miss_idx[0][0])] = -1
                            real[real.index(miss_idx[0][0]) + 1] = -1
                        else:
                            real[real.index(miss_idx[0][0])] = -1
                            real[real.index(miss_idx[1][0])] = -1
                        chains_real.append(real)
                        chains_fake.append(fake)

        if h_officer_num > 2 or h_officer_num < 0:
            raise ValueError(f"heart office count {h_officer_num} is illegal.")
        return chains_real, chains_fake

    def playable_cards_from_hand(self, player_group_id, player_cards_list):
        """ Get playable cards from hand  # 查找当前可以出的牌
        Returns:
            set: set of string of playable cards
        """
        h_officer_num = self.count_heart_officer(player_group_id, player_cards_list)  # 计算红心参谋

        current_hand = cards2str(player_cards_list)
        cards_dict = collections.defaultdict(int)
        for card in current_hand:
            cards_dict[card] += 1
        cards_count = np.array([cards_dict[k] for k in CARD_RANK_STR])
        playable_cards = set()

        non_zero_indexes = np.argwhere(cards_count > 0)  # 当前有的牌
        more_than_1_indexes = np.argwhere(cards_count > 1)  # 大于一张的牌
        more_than_2_indexes = np.argwhere(cards_count > 2)  # 大于两张的牌
        more_than_3_indexes = np.argwhere(cards_count > 3)  # 大于三张的牌

        # 1. Solo
        for i in non_zero_indexes:
            playable_cards.add((CARD_RANK_STR[i[0]], CARD_RANK_STR[i[0]]))

        # 2. Pair
        for i in more_than_1_indexes:
            playable_cards.add((CARD_RANK_STR[i[0]] * 2, CARD_RANK_STR[i[0]] * 2))
        if h_officer_num >= 1:  # 参谋可以配任意一张非自己本身存在的牌。
            for i in non_zero_indexes:
                if CARD_RANK_STR.index(self.officer[player_group_id]) > i[0]:
                    playable_cards.add((CARD_RANK_STR[i[0]] + self.officer[player_group_id], CARD_RANK_STR[i[0]] * 2))
                elif CARD_RANK_STR.index(self.officer[player_group_id]) == i[0]:
                    if h_officer_num > 1:
                        assert h_officer_num == 2, "Num of h_officer is invalid!"
                        playable_cards.add((CARD_RANK_STR[i[0]] + self.officer[player_group_id], CARD_RANK_STR[i[0]] * 2))
                else:
                    playable_cards.add((self.officer[player_group_id] + CARD_RANK_STR[i[0]], CARD_RANK_STR[i[0]] * 2))

        # 3. bomb
        for i in more_than_3_indexes:
            for j in range(4, cards_count[i[0]] + 1):
                cards = CARD_RANK_STR[i[0]] * j
                playable_cards.add((cards, cards))
        if h_officer_num == 1:
            for i in more_than_2_indexes:
                for j in range(3, cards_count[i[0]] + 1):
                    if CARD_RANK_STR.index(self.officer[player_group_id]) > i[0]:
                        cards = CARD_RANK_STR[i[0]] * j + self.officer[player_group_id]
                        playable_cards.add((cards, CARD_RANK_STR[i[0]] * (j + 1)))
                    elif CARD_RANK_STR.index(self.officer[player_group_id]) == i[0]:
                        # 等于的情况在上面more_than_3_indexes已处理掉了
                        pass
                    else:
                        cards = self.officer[player_group_id] + CARD_RANK_STR[i[0]] * j
                        playable_cards.add((cards, CARD_RANK_STR[i[0]] * (j + 1)))
        if h_officer_num == 2:
            for i in more_than_1_indexes:
                for j in range(2, cards_count[i[0]] + 1):
                    if CARD_RANK_STR.index(self.officer[player_group_id]) > i[0]:
                        cards = CARD_RANK_STR[i[0]] * j + self.officer[player_group_id] * 2
                        playable_cards.add((cards, CARD_RANK_STR[i[0]] * (j + 2)))
                    elif CARD_RANK_STR.index(self.officer[player_group_id]) == i[0]:
                        pass
                    else:
                        cards = self.officer[player_group_id] * 2 + CARD_RANK_STR[i[0]] * j
                        playable_cards.add((cards, CARD_RANK_STR[i[0]] * (j + 2)))

        # 4. solo_chain_5
        solo_chain_real, solo_chain_fake = self.serial_chain_indexes(
            card_dict=cards_dict,
            h_officer_num=h_officer_num,
            office_str=self.officer[player_group_id],
            repeat_num=1,
            serial_len=5
        )
        for real, fake in zip(solo_chain_real, solo_chain_fake):
            playable_cards.add((
                self.card_list2str(real, self.officer[player_group_id]),
                self.card_list2str(fake, self.officer[player_group_id])
            ))

        # 5. pair_chain_3
        pair_chain_real, pair_chain_fake = self.serial_chain_indexes(
            card_dict=cards_dict,
            h_officer_num=h_officer_num,
            office_str=self.officer[player_group_id],
            repeat_num=2,
            serial_len=3
        )
        for real, fake in zip(pair_chain_real, pair_chain_fake):
            playable_cards.add((
                self.card_list2str(real, self.officer[player_group_id]),
                self.card_list2str(fake, self.officer[player_group_id])
            ))

        # 6. trio and trio_pair
        for i in more_than_2_indexes:
            # trio
            playable_cards.add((CARD_RANK_STR[i[0]] * 3, CARD_RANK_STR[i[0]] * 3))
            # trio_pair
            for j in more_than_1_indexes:
                if j < i:
                    playable_cards.add((CARD_RANK_STR[j[0]] * 2 + CARD_RANK_STR[i[0]] * 3, CARD_RANK_STR[j[0]] * 2 + CARD_RANK_STR[i[0]] * 3))
                elif j > i:
                    playable_cards.add((CARD_RANK_STR[i[0]] * 3 + CARD_RANK_STR[j[0]] * 2, CARD_RANK_STR[i[0]] * 3 + CARD_RANK_STR[j[0]] * 2))
        if h_officer_num == 2:
            for i in non_zero_indexes:
                playable_cards.add((CARD_RANK_STR[i[0]] + self.officer[player_group_id] * 2, CARD_RANK_STR[i[0]] * 3))
                for j in more_than_1_indexes:
                    if j < i:
                        playable_cards.add((CARD_RANK_STR[i[0]] + self.officer[player_group_id] * 2 + CARD_RANK_STR[i[0]] * 3, CARD_RANK_STR[i[0]] * 3 + CARD_RANK_STR[i[0]] * 3))
                    elif j > i:
                        playable_cards.add((CARD_RANK_STR[i[0]] + self.officer[player_group_id] * 2 + CARD_RANK_STR[j[0]] * 2, CARD_RANK_STR[i[0]] * 3 + CARD_RANK_STR[j[0]] * 2))

            for i in more_than_1_indexes:
                playable_cards.add((CARD_RANK_STR[i[0]] * 2 + self.officer[player_group_id], CARD_RANK_STR[i[0]] * 3))
                for j in more_than_1_indexes:
                    if j < i:
                        playable_cards.add((CARD_RANK_STR[i[0]] * 2 + self.officer[player_group_id] + CARD_RANK_STR[i[0]] * 3, CARD_RANK_STR[i[0]] * 3 + CARD_RANK_STR[i[0]] * 3))
                    elif j > i:
                        playable_cards.add((CARD_RANK_STR[i[0]] * 2 + self.officer[player_group_id] + CARD_RANK_STR[j[0]] * 2, CARD_RANK_STR[i[0]] * 3 + CARD_RANK_STR[j[0]] * 2))

        # trio_chain_2
        # 7. trio_chain_2
        trio_chain_real, pair_chain_fake = self.serial_chain_indexes(
            card_dict=cards_dict,
            h_officer_num=h_officer_num,
            office_str=self.officer[player_group_id],
            repeat_num=3,
            serial_len=2
        )
        for real, fake in zip(pair_chain_real, pair_chain_fake):
            playable_cards.add((
                self.card_list2str(real, self.officer[player_group_id]),
                self.card_list2str(fake, self.officer[player_group_id])
            ))

        # 8. rocket  王炸
        if cards_count[13] == 2 and cards_count[14] == 2:
            playable_cards.add((CARD_RANK_STR[13] * 2 + CARD_RANK_STR[14] * 2, CARD_RANK_STR[13] * 2 + CARD_RANK_STR[14] * 2))

        playable_cards_post_process = [i[1] for i in list(playable_cards)]
        # return playable_cards
        return playable_cards_post_process

    def card_list2str(self, card_list, office_str):
        res = ""
        for card in card_list:
            if card == -1:
                res += office_str
            else:
                res += INDEX_CARD[card]
        return res

    # # 重新计算当前可以出的牌型
    def calc_playable_cards(self, player):
        """
        Recalculate all legal cards the player can play according to his
        current hand.
        Args:
            player (Player object): object of Player
            init_flag (boolean): For the first time, set it True to accelerate
              the preocess.
        Returns:
            list: list of string of playable cards
        """
        removed_playable_cards = []

        player_id = player.player_id
        player_group_id = player.group_id
        # 当前手上的牌
        current_hand = cards2str(player.current_hand)
        h_officer_num = self.count_heart_officer(player_group_id, player.current_hand)
        missed = None
        #
        for single in player.singles:
            if single not in current_hand:
                missed = single
                break

        playable_cards = self.playable_cards[player_id].copy()
        # print(playable_cards)
        # 有没有的牌面值
        if missed is not None:
            position = player.singles.find(missed)
            player.singles = player.singles[position + 1:]
            for cards in playable_cards:
                # 如果当前缺少某张牌或者当前没有对应牌型
                if missed in cards or (
                        not contains_cards(current_hand, cards, self.officer[player_group_id], h_officer_num, player.current_hand)):
                    # 移除可出牌型
                    removed_playable_cards.append(cards)
                    self.playable_cards[player_id].remove(cards)
        # 没有缺失的牌
        else:
            for cards in playable_cards:
                if not contains_cards(current_hand, cards, self.officer[player_group_id], h_officer_num, player.current_hand):
                    # del self.playable_cards[player_id][cards]
                    removed_playable_cards.append(cards)
                    self.playable_cards[player_id].remove(cards)
        # 移除的可出牌型
        self._recorded_removed_playable_cards[player_id].append(removed_playable_cards)
        # print("2", self.playable_cards[player_id])
        return self.playable_cards[player_id]

    # 获取当前玩家可出的牌
    def get_playable_cards(self, player):
        """ Provide all legal cards the player can play according to his
        current hand.
        Args:
            player (Player object): object of Player
            init_flag (boolean): For the first time, set it True to accelerate
              the preocess.
        Returns:
            list: list of string of playable cards
        """
        return self.playable_cards[player.player_id]

    # 判断当前玩家是否出完牌
    @staticmethod
    def judge_game(players, player_id):
        """
        Args:
            players (list): list of Player objects
            player_id (int): integer of player's id
        """
        player = players[player_id]
        if len(player.current_hand) == 0:
            return True
        return False

    @staticmethod
    def judge_payoffs(winner_id):
        payoffs = np.array([0, 0, 0, 0])
        # 双上
        if len(winner_id) == 2:
            payoffs[winner_id[0]] = 3
            payoffs[winner_id[1]] = 3
        elif len(winner_id) == 3:
            # 1、3
            if winner_id[0] % 2 == winner_id[2] % 2:
                payoffs[winner_id[0]] = 2
                payoffs[winner_id[2]] = 2
            # 1、4
            else:
                payoffs[winner_id[0]] = 1
                for i in range(4):
                    if i not in winner_id:
                        payoffs[i] = 1
        return payoffs

    def count_heart_officer(self, player_group_id, cards_list):
        """
        统计红心参谋的数量
        """
        cnt = 0
        for card in cards_list:
            if card.rank == self.officer[player_group_id] and card.suit == 'H':
                cnt += 1
        return cnt
