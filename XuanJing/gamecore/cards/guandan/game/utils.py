import os
import json

import threading
import numpy as np
from collections import OrderedDict
import collections

import XuanJing

class Card:
    """
    Card stores the suit and rank of a single card
    Note:
        The suit variable in a standard card game should be one of [S, H, D, C, BJ, RJ]
        meaning [Spades, Hearts, Diamonds, Clubs, Black Joker, Red Joker],
        Similarly the rank variable should be one of [A, 2, 3, 4, 5, 6, 7, 8, 9, T, J, Q, K].
    """
    suit = None
    rank = None
    valid_suit = ['S', 'H', 'D', 'C', 'BJ', 'RJ']
    valid_rank = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']

    def __init__(self, suit, rank):
        ''' Initialize the suit and rank of a card

        Args:
            suit: string, suit of the card, should be one of valid_suit
            rank: string, rank of the card, should be one of valid_rank
        '''
        self.suit = suit
        self.rank = rank

    def __eq__(self, other):
        if isinstance(other, Card):
            return self.rank == other.rank and self.suit == other.suit
        else:
            # don't attempt to compare against unrelated types
            return NotImplemented

    def __hash__(self):
        suit_index = Card.valid_suit.index(self.suit)
        rank_index = Card.valid_rank.index(self.rank)
        return rank_index + 100 * suit_index

    def __str__(self):
        ''' Get string representation of a card.

        Returns:
            string: the combination of rank and suit of a card. Eg: AS, 5H, JD, 3C, ...
        '''
        return self.rank + self.suit

    def get_index(self):
        ''' Get index of a card.

        Returns:
            string: the combination of suit and rank of a card. Eg: 1S, 2H, AD, BJ, RJ...
        '''
        return self.suit+self.rank


def set_seed(seed):
    if seed is not None:
        import subprocess
        import sys

        reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
        installed_packages = [r.decode().split('==')[0] for r in reqs.split()]
        if 'torch' in installed_packages:
            import torch
            torch.backends.cudnn.deterministic = True
            torch.manual_seed(seed)
        np.random.seed(seed)
        import random
        random.seed(seed)


def get_device():
    import torch
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("--> Running on the GPU")
    else:
        device = torch.device("cpu")
        print("--> Running on the CPU")

    return device

# Read required docs
ROOT_PATH = XuanJing.gamecore.cards.guandan.game.__path__[0]

# a map of card to its type. Also return both dict and list to accelerate
with open(os.path.join(ROOT_PATH, 'card_type.json'), 'r') as file:
    data = json.load(file, object_pairs_hook=OrderedDict)
    CARD_TYPE = (data, list(data), set(data))

# a map of type to its cards
with open(os.path.join(ROOT_PATH, 'type_card.json'), 'r') as file:
    TYPE_CARD = json.load(file, object_pairs_hook=OrderedDict)

# 获得比之前玩家出的牌更大的牌
def get_gt_cards(player, greater_player, officer, h_officer_num):
    """
    Provide player's cards which are greater than the ones played by
    previous player in one round

    Args:
        player (Player object): the player waiting to play cards
        greater_player (Player object): the player who played current biggest cards.

    Returns:
        list: list of string of greater cards

    Note:
        1. return value contains 'pass'
    """
    # add 'pass' to legal actions
    gt_cards = []
    if len(greater_player.current_hand) > 0:
        gt_cards = ['pass']

    target_cards = greater_player.played_cards
    if isinstance(target_cards, tuple):
        target_types = CARD_TYPE[0][target_cards[1]]
    else:
        target_types = CARD_TYPE[0][target_cards]
    type_dict = {}
    for card_type, weight in target_types:
        if card_type not in type_dict:
            type_dict[card_type] = weight

    # 如果上个玩家出四大天王，没有牌比它大
    if 'rocket' in type_dict:
        return gt_cards

    # 炸弹
    type_dict['rocket'] = -1

    for i in range(11, 4):
        if i == 5:
            if "straight_flush" not in type_dict:
                type_dict["straight_flush"] = -1
            else:
                break
        if "bomb_" + str(i) not in type_dict:
            type_dict["bomb_" + str(i)] = -1
        else:
            break
    current_hand = cards2str(player.current_hand)
    for card_type, weight in type_dict.items():
        candidate = TYPE_CARD[card_type]
        for can_weight, cards_list in candidate.items():
            if int(can_weight) > int(weight):
                for cards in cards_list:
                    if cards not in gt_cards and contains_cards(current_hand, cards, officer, h_officer_num,
                                                                player.current_hand):
                        # if self.contains_cards(current_hand, cards):
                        gt_cards.append(cards)
    return gt_cards


# 按照牌的大小排序
def sort_card(card_1, card_2):
    """ Compare the rank of two cards of Card object

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


def init_standard_deck():
    ''' Initialize a standard deck of 52 cards

    Returns:
        (list): A list of Card object
    '''
    suit_list = ['S', 'H', 'D', 'C']
    rank_list = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
    res = [Card(suit, rank) for suit in suit_list for rank in rank_list]
    return res


def init_54_deck():
    ''' Initialize a standard deck of 52 cards, BJ and RJ

    Returns:
        (list): Alist of Card object
    '''
    suit_list = ['S', 'H', 'D', 'C']
    rank_list = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
    res = [Card(suit, rank) for suit in suit_list for rank in rank_list]
    res.append(Card('BJ', ''))
    res.append(Card('RJ', ''))
    return res


def init_guandan_deck():
    ''' Initialize a standard deck of 52 cards, BJ and RJ

    Returns:
        (list): Alist of Card object
    '''
    suit_list = ['S', 'H', 'D', 'C']
    rank_list = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    res = []
    for i in range(2):
        res += [Card(suit, rank) for suit in suit_list for rank in rank_list]
        res.append(Card('BJ', ''))
        res.append(Card('RJ', ''))
    return res


def rank2int(rank):
    ''' Get the coresponding number of a rank.

    Args:
        rank(str): rank stored in Card object

    Returns:
        (int): the number corresponding to the rank

    Note:
        1. If the input rank is an empty string, the function will return -1.
        2. If the input rank is not valid, the function will return None.
    '''
    if rank == '':
        return -1
    elif rank.isdigit():
        if int(rank) >= 2 and int(rank) <= 10:
            return int(rank)
        else:
            return None
    elif rank == 'A':
        return 14
    elif rank == 'T':
        return 10
    elif rank == 'J':
        return 11
    elif rank == 'Q':
        return 12
    elif rank == 'K':
        return 13
    return None


def elegent_form(card):
    ''' Get a elegent form of a card string

    Args:
        card (string): A card string

    Returns:
        elegent_card (string): A nice form of card
    '''
    suits = {'S': '♠', 'H': '♥', 'D': '♦', 'C': '♣', 's': '♠', 'h': '♥', 'd': '♦', 'c': '♣'}
    rank = '10' if card[1] == 'T' else card[1]

    return suits[card[0]] + rank


def remove_illegal(action_probs, legal_actions):
    ''' Remove illegal actions and normalize the
        probability vector

    Args:
        action_probs (numpy.array): A 1 dimention numpy array.
        legal_actions (list): A list of indices of legal actions.

    Returns:
        probd (numpy.array): A normalized vector without legal actions.
    '''
    # print(action_probs,legal_actions)
    probs = np.zeros(action_probs.shape[0])
    probs[legal_actions] = action_probs[legal_actions]
    if np.sum(probs) == 0:
        probs[legal_actions] = 1 / len(legal_actions)
    else:
        probs /= sum(probs)
    return probs


def tournament(env, num):
    ''' Evaluate he performance of the agents in the environment

    Args:
        env (Env class): The environment to be evaluated.
        num (int): The number of games to play.

    Returns:
        A list of avrage payoffs for each player
    '''
    payoffs = [0 for _ in range(env.num_players)]
    win_probs = [[0, 0, 0, 0] for _ in range(env.num_players)]
    counter = 0
    while counter < num:
        _, _payoffs = env.run(is_training=False)
        # print("payoff", _payoffs)
        if isinstance(_payoffs, list):
            for _p in _payoffs:
                for i, _ in enumerate(payoffs):
                    payoffs[i] += _p[i]
                counter += 1
        else:
            for i, _ in enumerate(payoffs):
                payoffs[i] += _payoffs[i]
            counter += 1
        winner_id = env.game.winner_id
        for i in range(len(winner_id)):
            win_probs[winner_id[i]][i] += 1
    for i, _ in enumerate(payoffs):
        payoffs[i] /= counter
    for i in range(len(win_probs)):
        for j in range(len(win_probs[i])):
            win_probs[i][j] /= counter
    return payoffs,win_probs


def cards2str(cards):
    """
    Get the corresponding string representation of cards

    Args:
        cards (list): list of Card objects

    Returns:
        string: string representation of cards
    """
    response = ''
    for card in cards:
        if card.rank == '':
            response += card.suit[0]
        else:
            response += card.rank
    return response

class LocalObjs(threading.local):
    def __init__(self):
        self.cached_candidate_cards = None


_local_objs = LocalObjs()


def contains_cards(candidate, target, officer, h_officer_num, cards_list):
    """
    Check if cards of candidate contains cards of target.

    Args:
        candidate (string): A string representing the cards of candidate
        target (string): A string representing the number of cards of target

    Returns:
        boolean
    """
    # In normal cases, most continuous calls of this function
    #   will test different targets against the same candidate.
    # So the cached counts of each card in candidate can speed up
    #   the comparison for following tests if candidate keeps the same.
    if not _local_objs.cached_candidate_cards or _local_objs.cached_candidate_cards != candidate:
        _local_objs.cached_candidate_cards = candidate
        cards_dict = collections.defaultdict(int)
        # 当前所有牌
        for card in candidate:
            cards_dict[card] += 1
        _local_objs.cached_candidate_cards_dict = cards_dict
    cards_dict = _local_objs.cached_candidate_cards_dict
    # 如果目标牌型为空
    if target == '':
        return True

    curr_card = target[0]
    # 第一张牌
    # for i in range(len(target)):
    #     if target[i] != officer:
    #         curr_card = target[i]
    #         break
    curr_count = 0
    for card in target:
        # if card == officer:
        #     h_officer_num -= 1
        #     if h_officer_num < 0:
        #         return False
        #     continue
        if card != curr_card:
            if cards_dict[curr_card] < curr_count:
                return False
            curr_card = card
            curr_count = 1
        else:
            curr_count += 1
    if cards_dict[curr_card] < curr_count:
        return False
    return True


def cards2str_with_suit(cards):
    ''' Get the corresponding string representation of cards with suit

    Args:
        cards (list): list of Card objects

    Returns:
        string: string representation of cards
    '''
    return ' '.join([card.suit + card.rank for card in cards])


CARD_RANK_STR = [
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9',
    'T',
    'J',
    'Q',
    'K',
    'A',
    'B',
    'R'
]

CARD_RANK_STR_INDEX = {
    '2': 0,
    '3': 1,
    '4': 2,
    '5': 3,
    '6': 4,
    '7': 5,
    '8': 6,
    '9': 7,
    'T': 8,
    'J': 9,
    'Q': 10,
    'K': 11,
    'A': 12,
    'B': 13,
    'R': 14
}

INDEX_CARD = {
    0: '2',
    1: '3',
    2: '4',
    3: '5',
    4: '6',
    5: '7',
    6: '8',
    7: '9',
    8: 'T',
    9: 'J',
    10: 'Q',
    11: 'K',
    12: 'A',
    13: 'B',
    14: 'R'
}

CARD_RANK = [
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9',
    'T',
    'J',
    'Q',
    'K',
    'A',
    'BJ',
    'RJ'
]
