from random import shuffle
import numpy as np


class Node(object):
    def __init__(self, key, actionDict, nActions=2):
        """
        """
        self.key = key
        self.nActions = nActions  # 可选动作个数，默认为2个
        self.regretSum = np.zeros(self.nActions)  # 遗憾值记录列表
        self.strategySum = np.zeros(self.nActions)  # 策略记录列表
        self.actionDict = actionDict
        self.strategy = np.repeat(1 / self.nActions, self.nActions)  # 初始化策略

        self.reachPr = 0  # 到达概率
        self.reachPrSum = 0

    def updateStrategy(self):
        self.strategySum += self.reachPr * self.strategy
        self.reachPrSum += self.reachPr

        self.strategy = self.getStrategy()  # 更新策略
        self.reachPr = 0

    def getStrategy(self):
        regrets = self.regretSum
        regrets[regrets < 0] = 0
        normalizingSum = sum(regrets)
        if normalizingSum > 0:
            return regrets / normalizingSum
        else:
            return np.repeat(1 / self.nActions, self.nActions)

    def getAverageStrategy(self):
        strategy = self.strategySum / self.reachPrSum
        # Re-normalize
        total = sum(strategy)
        strategy /= total
        return strategy

    def __str__(self):
        strategies = ['{:03.2f}'.format(x)
                      for x in self.getAverageStrategy()]
        return '{} {}'.format(self.key.ljust(6), strategies)






def displayResults(ev, i_map):
    print('player 1 expected value: {}'.format(ev))
    print('player 2 expected value: {}'.format(-1 * ev))

    print()
    print('player 1 strategies:')
    sorted_items = sorted(i_map.items(), key=lambda x: x[0])
    for _, v in filter(lambda x: len(x[0]) % 2 == 0, sorted_items):
        print(v)
    print()
    print('player 2 strategies:')
    for _, v in filter(lambda x: len(x[0]) % 2 == 1, sorted_items):
        print(v)


class Kunh(object):
    def __init__(self):
        self.nodeMap = {}  # 创建节点哈希表
        self.deck = np.array([0, 1, 2])
        self.nAction = 2

    def train(self, nIterations=50000):
        expectedGameValue = 0
        for _ in range(nIterations):
            shuffle(self.deck)  # 打乱牌序
            expectedGameValue += self.cfr('', 1, 1)
            for _, v in self.nodeMap.items():
                v.updateStrategy()

        expectedGameValue /= nIterations
        displayResults(expectedGameValue, self.nodeMap)

    def cfr(self, history, pr_1, pr_2):
        """
        """
        n = len(history)
        isPlayer1 = n % 2 == 0  # 判断是否是玩家1

        playerCard = self.deck[0] if isPlayer1 else self.deck[1]

        if self.isTerminal(history):
            cardPlayer = self.deck[0] if isPlayer1 else self.deck[1]
            cardOpponent = self.deck[1] if isPlayer1 else self.deck[0]
            reward = self.getReward(history, cardPlayer, cardOpponent)
            return reward

        node = self.getNode(playerCard, history)
        strategy = node.strategy

        # 对于每个动作的遗憾收益
        actionUtils = np.zeros(self.nAction)

        for act in range(self.nAction):
            nextHistory = history + node.actionDict[act]  # 添加历史动作
            if isPlayer1:
                actionUtils[act] = -1 * self.cfr(nextHistory, pr_1 * strategy[act], pr_2)  # 收益等于对手收益 * -1
            else:
                actionUtils[act] = -1 * self.cfr(nextHistory, pr_1, pr_2 * strategy[act])

        # Utility of information set
        util = sum(actionUtils * strategy)
        regrets = actionUtils - util

        if isPlayer1:
            node.reachPr += pr_1  # 更新节点的到达概率和遗憾值和
            node.regretSum += pr_2 * regrets
        else:
            node.reachPr += pr_2
            node.regretSum += pr_1 * regrets
        return util

    @staticmethod
    def isTerminal(history):
        """p表示pass，b表示bet"""
        if history[-2:] == 'pp' or history[-2:] == 'bb' or history[-2:] == 'bp':
            return True

    @staticmethod
    def getReward(history, playerCard, opponentCard):
        """"""
        terminalPass = history[-1] == 'p'
        doubleBet = history[-2:] == 'bb'
        if terminalPass:  # 如果最后一个状态是pass。
            if history[-2:] == 'pp':  # 都pass则比较大小。
                return 1 if playerCard > opponentCard else -1
            else:  # bet , pass, 则我方赢
                return 1
        elif doubleBet:
            return 2 if playerCard > opponentCard else -2

    def getNode(self, card, history):
        key = str(card) + " " + history
        if key not in self.nodeMap:
            actionDict = {0: 'p', 1: 'b'}
            infoSet = Node(key, actionDict)
            self.nodeMap[key] = infoSet
            return infoSet
        return self.nodeMap[key]

if __name__ == "__main__":
    trainer = Kunh()
    trainer.train(nIterations = 25000)