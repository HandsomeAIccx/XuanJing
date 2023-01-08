
from XuanJing.gamecore.cards.guandan.env import Env
from XuanJing.gamecore.cards.guandan.random_agent import RandomAgent
from XuanJing.gamecore.cards.guandan.pygame_agent import PyGameAgent

if __name__ == "__main__":
    agents = [
        RandomAgent(),
        # PyGameAgent(),
        RandomAgent(),
        RandomAgent(),
        RandomAgent(),
    ]
    env = Env(agents)

    _, _payoffs = env.run()
