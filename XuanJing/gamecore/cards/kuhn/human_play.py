from XuanJing.gamecore.cards.kuhn.kuhn import KuhnPoker

if __name__ == "__main__":
    env = KuhnPoker()
    done, reward, obs = env.reset()
    print("Start New Game!")
    for i in range(10000):
        env.render()
        action = int(input())
        done, reward, obs = env.step(action)
        if done:
            print("Game Done! Reward is : ", reward)
            env.reset()
            print("Start New Game!")