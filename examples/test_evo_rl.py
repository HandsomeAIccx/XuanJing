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

"""
"Implementation of "Evolution Strategies as a Scalable Alternative to Reinforcement Learning"
ref: https://github.com/hoang-tn-nguyen/Evolutionary-Strategies
"""

from XuanJing.env.build_env import env_vector
from XuanJing.env.sample.sampler import Sampler
from XuanJing.actor.actor_group.es_actor import EsActor
from XuanJing.enhancement.advantage import enhance_advantage

# # --- Core Modules ---
# class Expectation(nn.Module):
#     def __init__(self):
#         '''
#         Input:
#             F: Values of F(x)
#             P: Distribution P(x)
#         Return:
#             E[F(x)] = \sum{F(x).P(x)}
#             => E[F] = \sum{F.P}
#         '''
#         super().__init__()
#
#     def forward(self, F, P):
#         return (F * P).sum()
#
#
# class NormalAutograd(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, mu, sigma, size, random_state):
#         eps = torch.randn(size, *mu.shape, device=mu.device, generator=random_state)
#         ctx.save_for_backward(eps, sigma)
#         theta = mu + sigma * eps
#         ratio = torch.ones(len(eps), dtype=torch.float32, device=eps.device) / len(eps)
#         ctx.mark_non_differentiable(theta)
#         return theta, ratio
#
#     @staticmethod
#     def backward(ctx, grad_theta, grad_ratio):
#         eps, sigma = ctx.saved_tensors
#         grad_mu, grad_sigma = None, None
#         if ctx.needs_input_grad[0]:
#             grad_mu = grad_ratio @ eps / sigma
#         if ctx.needs_input_grad[1]:
#             grad_sigma = grad_ratio @ (eps ** 2) / sigma
#         return (grad_mu, grad_sigma, None, None)
#
#
# class Normal(nn.Module):
#     def __init__(self, device='cpu', seed=0):
#         super().__init__()
#         self.device = device
#         self.random_state = torch.Generator(device=self.device).manual_seed(seed)
#
#     def forward(self, mu, sigma, size):
#         '''
#         Input:
#             mu: torch.tensor: Mean of the distribution
#             sigma: torch.tensor: Standard deviation of the distribution
#             size: int: Number of samples to be drawn from the distribution
#         Return:
#             theta: torch.tensor: Samples drawn from the distribution (N,*mu.shape)
#             ratio: torch.tensor: 1 / size (N)
#         '''
#         theta, ratio = NormalAutograd.apply(mu, sigma, size, self.random_state)
#         return theta, ratio
#
#
# def normalize(input, eps=1e-9):
#     return (input - input.mean()) / (input.std() + eps)


class OpenAiES:
    def __init__(self, model, learning_rate, noise_std, noise_decay=1.0, lr_decay=1.0, decay_step=50, norm_rewards=True):
        self.model = model

        self._lr = learning_rate
        self._noise_std = noise_std

        self.noise_decay = noise_decay
        self.lr_decay = lr_decay
        self.decay_step = decay_step

        self.norm_rewards = norm_rewards

        self._population = None
        self._count = 0

    @property
    def noise_std(self):
        step_decay = np.power(self.noise_decay, np.floor((1 + self._count) / self.decay_step))

        return self._noise_std * step_decay

    @property
    def lr(self):
        step_decay = np.power(self.lr_decay, np.floor((1 + self._count) / self.decay_step))

        return self._lr * step_decay

    def generate_population(self, npop=50):
        self._population = []

        for i in range(npop):
            new_model = deepcopy(self.model)
            new_model.E = []

            for i, layer in enumerate(new_model.W):
                noise = np.random.randn(layer.shape[0], layer.shape[1])

                new_model.E.append(noise)
                new_model.W[i] = new_model.W[i] + self.noise_std * noise
            self._population.append(new_model)

        return self._population

    def update_population(self, rewards):
        if self._population is None:
            raise ValueError("populations is none, generate & eval it first")

        # z-normalization (?) - works better, but slower
        if self.norm_rewards:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        for i, layer in enumerate(self.model.W):
            w_updates = np.zeros_like(layer)

            for j, model in enumerate(self._population):
                w_updates = w_updates + (model.E[i] * rewards[j])

            # SGD weights update
            self.model.W[i] = self.model.W[i] + (self.lr / (len(rewards) * self.noise_std)) * w_updates

        self._count = self._count + 1

    def get_model(self):
        return self.model

# def simulate_single(weights, env):
#     total_reward = 0.0
#     num_run = 10
#     for t in range(num_run):
#         observation = env.reset()
#         observation = torch.tensor(observation, dtype=torch.float32)
#         for i in range(300):
#             action = 1 if observation @ weights > 0 else 0
#             observation, reward, done, info = env.step(np.array(action))
#             observation = torch.tensor(observation, dtype=torch.float32)
#             total_reward += reward
#             if done:
#                 break
#     return total_reward / num_run

# def simulate(batch_weights, env):
#     rewards = []
#     for weights in batch_weights:
#         rewards.append(simulate_single(weights, env))
#     return torch.tensor(rewards, dtype=torch.float32)

CONTINUOUS_ENVS = ('LunarLanderContinuous', "MountainCarContinuous", "BipedalWalker")

def eval_policy(policy, env, n_steps=200):
    try:
        env_name = env.spec._env_name
    except AttributeError:
        env_name = env._env_name

    total_reward = 0

    obs = env.reset()
    for i in range(n_steps):
        if env_name in CONTINUOUS_ENVS:
            action = policy.predict(np.array(obs).reshape(1, -1), scale="tanh")
        else:
            action = policy.predict(np.array(obs).reshape(1, -1), scale="softmax")

        new_obs, reward, done, _ = env.step(action)

        total_reward = total_reward + reward
        obs = new_obs

        if done:
            break

    return total_reward


# for parallel
eval_policy_delayed = delayed(eval_policy)

def build_train(args):
    env = gym.make("CartPole-v0")
    # env = env_vector(args=args)

    actor_net = MLP(
        input_dim=int(np.prod(env.observation_space.shape)),
        output_dim=int(np.prod(env.action_space.n)),
        hidden_sizes=args.actor_net,
    )
    policy = ThreeLayerNetwork(
        in_features=int(np.prod(env.observation_space.shape)),
        out_features=int(np.prod(env.action_space.n)),
        hidden_sizes=args.actor_net
    )
    config = {}
    es = OpenAiES(
        model=policy,
        learning_rate=args.learning_rate,
        noise_std=args.noise_std,
        noise_decay=config.get("noise_decay", 1.0),
        lr_decay=config.get("lr_decay", 1.0),
        decay_step=config.get("decay_step", 50)
    )

    log = defaultdict(list)
    config["env_steps"] = 200
    for session in tqdm(range(args.n_sessions)):
        population = es.generate_population(args.population_size)
        rewards_jobs = (eval_policy_delayed(new_policy, env, config["env_steps"]) for new_policy in population)
        rewards = np.array(Parallel(n_jobs=4)(rewards_jobs))
        es.update_population(rewards)

        # populations stats
        log["pop_mean_rewards"].append(np.mean(rewards))
        log["pop_std_rewards"].append(np.std(rewards))

        # best policy stats
        if session % config.get("eval_step", 2) == 0:
            best_policy = es.get_model()

            best_rewards = np.zeros(10)
            for i in range(10):
                best_rewards[i] = eval_policy(best_policy, env, config["env_steps"])

            if True:
                # TODO: add timestamp
                print(f"Session: {session}")
                print(f"Mean reward: {round(np.mean(rewards), 4)}", f"std: {round(np.std(rewards), 3)}")
                print(f"lr: {round(es.lr, 5)}, noise_std: {round(es.noise_std, 5)}")

            log["best_mean_rewards"].append(np.mean(best_rewards))
            log["best_std_rewards"].append(np.std(best_rewards))

# actor = EsActor(actor_net, env, args)
#     #
#     optimizer = torch.optim.Adam(actor_net, lr=0.03)
#     #
#     sampler = Sampler(actor, env, args)
#
#     agent = ES(
#         actor_net,
#         optimizer,
#         args
#     )
#     expectation = Expectation()
#     for i_episode in range(args.num_episodes):
#         # theta, ratio = normal(mu, std, 50)
#         # sampler.sample_episode(n_episode=1)
#         score = simulate(agent.theta, env)
#         # avg_episode_reward = sampler.logging["Sample/avg_episode_reward"]
#         mean = expectation(normalize(-score), agent.ratio)
#         agent.update(mean)
#         print("step: {}, mean fitness: {:0.5}".format(i_episode, float(score.mean())))


def es_args():
    parser = argparse.ArgumentParser()
    # env
    parser.add_argument("--task", type=str, default="CartPole-v0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env_num", type=int, default=1)
    # agent
    parser.add_argument('--actor_net', type=list, default=[16, 16])
    parser.add_argument("--num_episodes", type=int, default=500)
    parser.add_argument("--n_sessions", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--noise_std", type=float, default=0.05)
    parser.add_argument("--population_size", type=int, default=512)

    args = parser.parse_known_args()[0]
    return args

if __name__ == "__main__":
    es_args = es_args()
    build_train(es_args)





