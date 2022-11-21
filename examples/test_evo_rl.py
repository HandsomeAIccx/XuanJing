import argparse

import gym
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm

"""
"Implementation of "Evolution Strategies as a Scalable Alternative to Reinforcement Learning"
ref: https://github.com/hoang-tn-nguyen/Evolutionary-Strategies
"""

from XuanJing.env.build_env import env_vector
from XuanJing.env.sample.sampler import Sampler
from XuanJing.actor.actor_group.es_actor import EsActor
from XuanJing.enhancement.advantage import enhance_advantage

# --- Core Modules ---
class Expectation(nn.Module):
    def __init__(self):
        '''
        Input:
            F: Values of F(x)
            P: Distribution P(x)
        Return:
            E[F(x)] = \sum{F(x).P(x)}
            => E[F] = \sum{F.P}
        '''
        super().__init__()

    def forward(self, F, P):
        return (F * P).sum()


class NormalAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mu, sigma, size, random_state):
        eps = torch.randn(size, *mu.shape, device=mu.device, generator=random_state)
        ctx.save_for_backward(eps, sigma)
        theta = mu + sigma * eps
        ratio = torch.ones(len(eps), dtype=torch.float32, device=eps.device) / len(eps)
        ctx.mark_non_differentiable(theta)
        return theta, ratio

    @staticmethod
    def backward(ctx, grad_theta, grad_ratio):
        eps, sigma = ctx.saved_tensors
        grad_mu, grad_sigma = None, None
        if ctx.needs_input_grad[0]:
            grad_mu = grad_ratio @ eps / sigma
        if ctx.needs_input_grad[1]:
            grad_sigma = grad_ratio @ (eps ** 2) / sigma
        return (grad_mu, grad_sigma, None, None)


class Normal(nn.Module):
    def __init__(self, device='cpu', seed=0):
        super().__init__()
        self.device = device
        self.random_state = torch.Generator(device=self.device).manual_seed(seed)

    def forward(self, mu, sigma, size):
        '''
        Input:
            mu: torch.tensor: Mean of the distribution
            sigma: torch.tensor: Standard deviation of the distribution
            size: int: Number of samples to be drawn from the distribution
        Return:
            theta: torch.tensor: Samples drawn from the distribution (N,*mu.shape)
            ratio: torch.tensor: 1 / size (N)
        '''
        theta, ratio = NormalAutograd.apply(mu, sigma, size, self.random_state)
        return theta, ratio


def normalize(input, eps=1e-9):
    return (input - input.mean()) / (input.std() + eps)


class ES(object):
    def __init__(
            self,
            actor_net,
            optimizer,
            args
    ):
        self.npop = 50  # population size
        self.actor_net = actor_net
        self.normal = Normal(actor_net[0].device)
        self.optimizer = optimizer
        self.theta, self.ratio = self.normal(actor_net[0], actor_net[1], self.npop)

    def update(self, mean):
        self.optimizer.zero_grad()
        mean.backward()
        self.optimizer.step()

        self.theta, self.ratio = self.normal(self.actor_net[0], self.actor_net[1], self.npop)


def simulate_single(weights, env):
    total_reward = 0.0
    num_run = 10
    for t in range(num_run):
        observation = env.reset()
        observation = torch.tensor(observation, dtype=torch.float32)
        for i in range(300):
            action = 1 if observation @ weights > 0 else 0
            observation, reward, done, info = env.step(np.array(action))
            observation = torch.tensor(observation, dtype=torch.float32)
            total_reward += reward
            if done:
                break
    return total_reward / num_run

def simulate(batch_weights, env):
    rewards = []
    for weights in batch_weights:
        rewards.append(simulate_single(weights, env))
    return torch.tensor(rewards, dtype=torch.float32)


def build_train(args):
    env = gym.make("CartPole-v0")
    # env = env_vector(args=args)

    mu = torch.randn(4, requires_grad=True)  # population mean
    std = torch.full(mu.shape, 0.5, requires_grad=True)  # population standard deviation

    actor_net = [mu, std]
    actor = EsActor(actor_net, env, args)
    #
    optimizer = torch.optim.Adam(actor_net, lr=0.03)
    #
    sampler = Sampler(actor, env, args)

    agent = ES(
        actor_net,
        optimizer,
        args
    )
    expectation = Expectation()
    for i_episode in range(args.num_episodes):
        # theta, ratio = normal(mu, std, 50)
        # sampler.sample_episode(n_episode=1)
        score = simulate(agent.theta, env)
        # avg_episode_reward = sampler.logging["Sample/avg_episode_reward"]
        mean = expectation(normalize(-score), agent.ratio)
        agent.update(mean)
        print("step: {}, mean fitness: {:0.5}".format(i_episode, float(score.mean())))


def es_args():
    parser = argparse.ArgumentParser()
    # env
    parser.add_argument("--task", type=str, default="CartPole-v0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env_num", type=int, default=1)
    # agent
    parser.add_argument("--num_episodes", type=int, default=500)

    args = parser.parse_known_args()[0]
    return args

if __name__ == "__main__":
    es_args = es_args()
    build_train(es_args)





