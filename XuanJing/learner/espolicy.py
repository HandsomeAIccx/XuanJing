import torch
import copy
from tqdm import tqdm

from XuanJing.learner.base import BaseLearner
from XuanJing.env.sample.sampler import Sampler
from XuanJing.enhancement.episode_reward import enhance_get_episode_reward


class PipeLearner(BaseLearner):
    @staticmethod
    def run(
            args,
            env,
            actor,
            agent
    ):
        sampler = Sampler(actor=actor, env=env, args=args)
        tqdm_range = tqdm(range(args.n_sessions))
        for session in tqdm_range:
            population_reward = torch.zeros(args.population_size)
            parameters, noise = agent.generate_population(args.population_size)
            for i in range(args.population_size):
                explore_net = copy.deepcopy(agent.actor_net)
                explore_net.from_vec(parameters[i])
                sampler.replace_actor(explore_net)
                episodes_patch = sampler.sample_episode(args.num_episodes)
                episodes_reward = enhance_get_episode_reward(episodes_patch)
                population_reward[i] = sum(episodes_reward) / len(episodes_reward)
            agent.update_net(population_reward)
            tqdm_range.set_postfix({"Generation": f"{session}",
                                    "AverageReward": f"{population_reward.mean()}",
                                    "BestReward": f"{max(population_reward)}"})