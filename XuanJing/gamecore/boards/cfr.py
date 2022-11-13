import numpy as np

from XuanJing.gamecore.boards.rps import RockPaperScissors


class CFRAgent(object):
    def __init__(self, env):
        self.regret_sum = {
            env.player_space[0]: np.zeros(env.action_space.n),
            env.player_space[1]: np.zeros(env.action_space.n)
        }

        self.strategy_sum = {
            env.player_space[0]: np.zeros(env.action_space.n),
            env.player_space[1]: np.zeros(env.action_space.n)
        }
        self.num_action = env.action_space.n
        self.possible_actions = np.arange(self.num_action)

    def take_action(self, game):
        player = game.current_player
        opponent = game.opponent_player

        strategy = self.regret2strategy(self.regret_sum[player])
        self.strategy_sum[player] += strategy
        player_action = np.random.choice(self.possible_actions, p=strategy)

        opp_strategy = self.regret2strategy(self.regret_sum[opponent])
        self.strategy_sum[opponent] += opp_strategy
        opp_action = np.random.choice(self.possible_actions, p=opp_strategy)

        player_reward = game.action_utility[player_action][opp_action]
        opp_reward = game.action_utility[opp_action][player_action]

        for a in range(self.num_action):
            player_regret = game.action_utility[a][opp_action] - player_reward
            opp_regret = game.action_utility[a][player_action] - opp_reward

            self.regret_sum[player][a] += player_regret
            self.regret_sum[opponent][a] += opp_regret

        return player_action

    def regret2strategy(self, regret_sum):
        """从遗憾值获取策略"""
        # Clip negative regrets for faster convergence
        clip_sum = np.clip(regret_sum, a_min=0, a_max=None)
        normalizing_sum = np.sum(clip_sum)
        if normalizing_sum > 0:
            clip_sum /= normalizing_sum
        else:
            clip_sum = np.repeat(1/self.num_action, self.num_action)
        return clip_sum

    def get_average_strategy(self, strategy_sum):
        average_strategy = [0, 0, 0]
        normalizing_sum = sum(strategy_sum)
        for a in range(self.num_action):
            if normalizing_sum > 0:
                average_strategy[a] = strategy_sum[0] / normalizing_sum
            else:
                average_strategy[a] = 1.0 / self.num_action
        return average_strategy


if __name__ == "__main__":
    env = RockPaperScissors()
    agent = CFRAgent(env)
    obs, done, reward = env.reset()
    for i in range(10000):
        env.render()
        action = agent.take_action(env)
        obs, done, reward = env.step(action)
        if done:
            print("reward is {}".format(reward))
            env.reset()

    print(agent.get_average_strategy(agent.strategy_sum[env.player_space[0]]))
    print(agent.get_average_strategy(agent.strategy_sum[env.player_space[1]]))