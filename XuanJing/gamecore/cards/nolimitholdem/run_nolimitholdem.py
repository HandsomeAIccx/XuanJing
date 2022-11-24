import random
import sys
if sys.version_info >= (3, 8):
    from typing import Literal, TypedDict  # pylint: disable=no-name-in-module
else:
    from typing_extensions import Literal, TypedDict
# import clubs
from typing import List, Optional, Union

class PokerConfig(TypedDict):
    num_players: int
    num_streets: int
    blinds: Union[int, List[int]]
    antes: Union[int, List[int]]
    raise_sizes: Union[
        int, Literal["pot", "inf"], List[Union[int, Literal["pot", "inf"]]]
    ]
    num_raises: Union[int, Literal["inf"], List[Union[int, Literal["inf"]]]]
    num_suits: int
    num_ranks: int
    num_hole_cards: int
    num_community_cards: Union[int, List[int]]
    num_cards_for_hand: int
    mandatory_num_hole_cards: int
    start_stack: int
    low_end_straight: bool
    order: Optional[List[str]]

NO_LIMIT_HOLDEM_SIX_PLAYER: PokerConfig = {
    "num_players": 6,
    "num_streets": 4,
    "blinds": [1, 2, 0, 0, 0, 0],
    "antes": 0,
    "raise_sizes": "inf",
    "num_raises": "inf",
    "num_suits": 4,
    "num_ranks": 13,
    "num_hole_cards": 2,
    "num_community_cards": [0, 3, 1, 1],
    "num_cards_for_hand": 5,
    "mandatory_num_hole_cards": 0,
    "start_stack": 200,
    "low_end_straight": True,
    "order": None,
}

from XuanJing.gamecore.cards.nolimitholdem.engine import Dealer


def main():
    # 1-2 no limit 6 player texas hold'em
    config = NO_LIMIT_HOLDEM_SIX_PLAYER
    dealer = Dealer(**config)
    obs = dealer.reset()

    while True:
        # number of chips a player must bet to call
        call = obs["call"]
        # smallest number of chips a player is allowed to bet for a raise
        min_raise = obs["min_raise"]

        rand = random.random()
        # 15% chance to fold
        if rand < 0.15:
            bet = 0
        # 80% chance to call
        elif rand < 0.95:
            bet = call
        # 5% to raise to min_raise
        else:
            bet = min_raise

        obs, rewards, done = dealer.step(bet)

        if all(done):
            break

    print(rewards)


if __name__ == "__main__":
    main()
