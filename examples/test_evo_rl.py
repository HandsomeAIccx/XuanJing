import argparse

import gym
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from XuanJing.utils.net.common import MLP
from XuanJing.utils.net.linear import ThreeLayerNetwork
from joblib import delayed
from joblib import Parallel
from copy import deepcopy
from collections import defaultdict
env = gym.make("CartPole-v0")
"""
"Implementation of "Evolution Strategies as a Scalable Alternative to Reinforcement Learning"
ref: https://github.com/hoang-tn-nguyen/Evolutionary-Strategies
"""

from XuanJing.env.build_env import env_vector
from XuanJing.env.sample.sampler import Sampler
from XuanJing.actor.actor_group.es_actor import EsActor
from XuanJing.enhancement.advantage import enhance_advantage

class EsAgent(object):
    def __init__(self,
                 actor_net):
        self.actor_net = actor_net
        self.sigma = 0.1
        self.lr = 0.01
        self.param_size = self.actor_net.to_vec().shape[0]

    def generate_population(self, population_size):
        parameters = self.actor_net.to_vec()
        noise = torch.randn(population_size, parameters.shape[0])
        springs = noise * self.sigma
        return springs + parameters, noise

    def update_actor_para(self, param):
        self.actor_net.from_vec(param)

    def update_net(self, adv, population_size, noise):
        parameters = self.actor_net.to_vec()
        parameters = parameters + self.lr / (population_size * self.sigma) * torch.matmul(noise.T, adv)
        self.actor_net.from_vec(parameters)
        self.lr *= 0.992354

def rewardFunction(param, args):
    nn = MLP(
        input_dim=int(np.prod(env.observation_space.shape)),
        output_dim=int(np.prod(env.action_space.n)),
        hidden_sizes=args.actor_net,
    )
    nn.from_vec(param)
    totalRewards=[]
    for i in range(10):
        done=False
        state=env.reset()
        rewards=0
        while not done:
            state = torch.tensor(state, dtype=torch.float)
            net_out = nn(state)
            action = np.argmax(net_out.detach().numpy())
            # action=NN.state2Value(state)
            nextState,reward,done,info=env.step(action)
            rewards+=reward
            state=nextState
        totalRewards.append(rewards)
    return sum(totalRewards)/len(totalRewards)

def build_train(args):

    # env = env_vector(args=args)

    actor_net = MLP(
        input_dim=int(np.prod(env.observation_space.shape)),
        output_dim=int(np.prod(env.action_space.n)),
        hidden_sizes=args.actor_net,
    )
    agent = EsAgent(
        actor_net
    )

    for session in range(args.n_sessions):
    # for session in tqdm(range(args.n_sessions)):
        reward = torch.zeros(args.population_size)
        noise = torch.randn(args.population_size, agent.param_size)
        springs = noise * 0.1
        parameters = agent.actor_net.to_vec()
        for i in range(args.population_size):
            spring = parameters + springs[i]
            reward[i] = rewardFunction(spring, args)
        advantage = (reward - reward.mean()) / reward.std()
        agent.update_net(advantage, args.population_size, noise)
        print(f"generation: {session} Average Reward: {reward.mean()} Best reward:{max(reward)}")



def es_args():
    parser = argparse.ArgumentParser()
    # env
    parser.add_argument("--task", type=str, default="CartPole-v0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env_num", type=int, default=1)
    # agent
    parser.add_argument('--actor_net', type=list, default=[128])
    parser.add_argument("--num_episodes", type=int, default=500)
    parser.add_argument("--n_sessions", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--noise_std", type=float, default=0.05)
    parser.add_argument("--population_size", type=int, default=50)

    args = parser.parse_known_args()[0]
    return args

if __name__ == "__main__":
    es_args = es_args()
    build_train(es_args)





