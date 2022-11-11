from XuanJing.gamecore.cards.blackjack.blackjack import BlackJackEnv

if __name__ == "__main__":
    env = BlackJackEnv()
    observation, info = env.reset(seed=42)

    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        print("observation", observation)
        if terminated or truncated:
            observation, info = env.reset()
    env.close()