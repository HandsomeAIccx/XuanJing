

import numpy as np
from numpy.random import choice
from XuanJing.gamecore.boards.rps import RockPaperScissors


class DepthLimitCFRAgent(object):
    def __init__(self, env):
        self.env = env
        self.regret_sum = np.zeros(env.action_space.n)
        self.strategy_sum = np.zeros(env.action_space.n)

        self.opp_regret_strategy = [[0, 0, 0]]
        self.opp_regret_sum = np.zeros(len(self.opp_regret_strategy))
        self.opp_strategy_sum = np.zeros(len(self.opp_regret_strategy))
        self.num_action = env.action_space.n
        self.possible_actions = np.arange(self.num_action)

    def reset_regrets(self):
        self.regret_sum = np.zeros(self.env.action_space.n)
        self.strategy_sum = np.zeros(self.env.action_space.n)
        self.opp_regret_sum = np.zeros(len(self.opp_regret_strategy))
        self.opp_strategy_sum = np.zeros(len(self.opp_regret_strategy))

    @staticmethod
    def regret2strategy(regret_sum):
        # Clip Negative Regrets For Faster Convergence.
        regret_sum = np.clip(regret_sum, a_min=0, a_max=None)
        normalizing_sum = np.sum(regret_sum)
        strategy = regret_sum
        if normalizing_sum > 0:
            strategy /= normalizing_sum
        else:
            strategy = np.repeat(1/len(regret_sum), len(regret_sum))
        return strategy

    def get_average_strategy(self, strategy_sum):
        average_strategy = [0, 0, 0]
        normalizing_sum = sum(strategy_sum)
        for a in range(self.num_action):
            if normalizing_sum > 0:
                average_strategy[a] = strategy_sum[a] / normalizing_sum
            else:
                average_strategy[a] = 1.0 / self.num_action
        return average_strategy

    def take_action(self, game):
        # Solve Current Sub Game.
        for i in range(5000):
            strategy = self.regret2strategy(self.regret_sum)
            opp_strategy = self.regret2strategy(self.opp_regret_sum)

            self.strategy_sum += strategy
            self.opp_strategy_sum += opp_strategy

            my_action = np.random.choice(np.arange(len(strategy)), p=strategy)
            opp_action = np.random.choice(np.arange(len(opp_strategy)), p=opp_strategy)

            # 玩家奖励是对手的遗憾值，对手的奖励是最小化遗憾值。
            my_reward = self.opp_regret_strategy[opp_action][my_action]
            opp_reward = -1 * self.opp_regret_strategy[opp_action][my_action]

            # Calculate regret for each of the strategies.
            for a in range(len(self.opp_regret_strategy)):
                # 对手新的遗憾值是，采取动作a的遗憾值减去拿到的即使奖励。
                self.opp_regret_sum[a] += (-1 * self.opp_regret_strategy[a][my_action]) - opp_reward
            for a in range(self.num_action):
                # 玩家的遗憾值是能够使得对手策略最大的那个动作，减去已经采取了的动作的差。
                self.regret_sum[a] += self.opp_regret_strategy[opp_action][a] - my_reward

        target_policy = self.get_average_strategy(self.strategy_sum)
        # Calculate Best Response
        values = [0 for _ in range(self.num_action)]
        for i in range(5000):
            hero_choice = np.random.choice(self.possible_actions, p=target_policy)
            villain_choice = np.random.choice(self.possible_actions)
            values[villain_choice] += game.action_utility[villain_choice][hero_choice]
        new_policy = self.regret2strategy(np.array(values, dtype=np.float64))

        new_policy_ev = [0 for _ in range(self.num_action)]
        for i in range(self.num_action):
            for j, p2_strategy in enumerate(new_policy):
                new_policy_ev[i] += p2_strategy * game.action_utility[i][j]

        # add best response to leaf node policies
        self.opp_regret_strategy.append(new_policy_ev)
        print(target_policy)
        self.reset_regrets()

        return my_action


if __name__ == "__main__":
    env = RockPaperScissors()
    agent = DepthLimitCFRAgent(env)
    env.reset()
    for i in range(10):
        # env.reset()
        # env.render()
        action = agent.take_action(env)
        # obs, done, reward = env.step(action)
        # if done:
        #     print("Game Done! Reward is : ", reward)
        #     env.reset()
        #     print("Start New Game!")
    # print(agent.)