import copy
import numpy as np

from XuanJing.gamecore.cards.kuhn.kuhn import KuhnPoker


class CFRAgent(object):
    def __init__(self):
        self.num_action = 2
        self.node_map = {}  # 创建结点哈希表
        self.possible_actions = np.arange(self.num_action)

    def take_action(self, game):
        self.cfr_search(game, 1, 1)
        for k, v in self.node_map.items():
            v.update_strategy()
        return np.random.choice(self.possible_actions, p=self.node_map[str(game)].strategy)

    def cfr_search(self, game, pr_1, pr_2):
        player = game.current_player
        opponent = game.opponent_player
        if game.done:
            return game.player_reward[player]

        node = self._get_node(str(game))
        strategy = node.strategy

        # Counterfactual utility per action
        action_utils = np.zeros(self.num_action)

        for act in range(self.num_action):
            sim_game = copy.deepcopy(game)
            sim_game.step(act)
            if player == "player1":
                action_utils = -1 * self.cfr_search(sim_game, pr_1 * strategy[act], pr_2)
            else:
                action_utils = -1 * self.cfr_search(sim_game, pr_1, pr_2 * strategy[act])

        # Utility of Information Set.
        util = sum(action_utils * strategy)
        regret = action_utils - util
        if player == "player1":
            node.reach_pr += pr_1
            node.regret_sum += pr_2 * regret
        else:
            node.reach_pr += pr_2
            node.regret_sum += pr_1 * regret
        return util

    def _get_node(self, key):
        if key not in self.node_map:
            instance_node = Node(key=key, num_action=self.num_action)
            self.node_map[key] = instance_node
        return self.node_map[key]


class Node(object):
    def __init__(self, key, num_action):
        self.key = key
        self.num_action = num_action
        self.regret_sum = np.zeros(self.num_action)
        self.strategy_sum = np.zeros(self.num_action)
        self.strategy = np.repeat(1 / self.num_action, self.num_action)
        self.reach_pr = 0
        self.reach_pr_sum = 0

    def update_strategy(self):
        self.strategy_sum += self.reach_pr * self.strategy
        self.reach_pr_sum += self.reach_pr
        self.strategy = self.regret2strategy(self.regret_sum)
        self.reach_pr = 0

    def regret2strategy(self, regret_sum):
        """从遗憾值获取策略"""
        # Clip negative regrets for faster convergence
        clip_sum = np.clip(regret_sum, a_min=0, a_max=None)
        normalizing_sum = np.sum(clip_sum)
        if normalizing_sum > 0:
            return clip_sum / normalizing_sum
        else:
            return np.repeat(1 / self.num_action, self.num_action)

    def get_average_strategy(self):
        strategy = self.strategy_sum / self.reach_pr_sum
        # Re-normalize
        total = sum(strategy)
        strategy /= total
        return strategy

    def __str__(self):
        strategies = ['{:03.2f}'.format(x)
                      for x in self.get_average_strategy()]
        return '{} {}'.format(self.key.ljust(6), strategies)


def display_results(i_map):
    # print('player 1 expected value: {}'.format(ev))
    # print('player 2 expected value: {}'.format(-1 * ev))

    print()
    print('player 1 strategies:')
    sorted_items = sorted(i_map.items(), key=lambda x: x[0])
    for _, v in filter(lambda x: len(x[0]) % 2 == 0, sorted_items):
        print(v)
    print()
    print('player 2 strategies:')
    for _, v in filter(lambda x: len(x[0]) % 2 == 1, sorted_items):
        print(v)


if __name__ == "__main__":
    env = KuhnPoker()
    agent = CFRAgent()
    done, reward, obs = env.reset()
    print("Start New Game!")
    for i in range(10000):
        env.render()
        action = agent.take_action(env)
        done, reward, obs = env.step(action)
        if done:
            print("Game Done! Reward is : ", reward)
            env.reset()
            print("Start New Game!")
    display_results(agent.node_map)