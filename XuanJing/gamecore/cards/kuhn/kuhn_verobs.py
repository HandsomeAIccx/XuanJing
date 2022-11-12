
class VetorKuhnObs(object):
    def __init__(self, env):
        self.env = env

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self):
        self.env.render()


if __name__ == "__main__":
    from XuanJing.gamecore.cards.kuhn.kuhn import KuhnPoker
    env = KuhnPoker()
    env_vecobs = VetorKuhnObs(env)
    done, reward, obs = env_vecobs.reset()
    print("Start New Game!")
    for i in range(10000):
        env_vecobs.render()
        action = int(input())
        done, reward, obs = env_vecobs.step(action)
        if done:
            print("Game Done! Reward is : ", reward)
            env_vecobs.reset()
            print("Start New Game!")