import torch.optim
import argparse
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import tensorboardX as tb

from XuanJing.utils.net.common import MLP
from XuanJing.env.sample.sampler import Sampler
from XuanJing.algorithms.modelfree.ppo import PPO
from XuanJing.enhancement.advantage import enhance_advantage
from XuanJing.actor.actor_group.softmax_actor import SoftmaxActor
from XuanJing.env.build_env import env_vector
from XuanJing.learner.base import BaseLearner


class PipeLearner(BaseLearner):
    @staticmethod
    def run(
            args
    ):
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        envs = env_vector(args=args)

        actor_net = MLP(
            input_dim=int(np.prod(envs.observation_space.shape)),
            output_dim=int(np.prod(envs.action_space.n)),
            hidden_sizes=args.actor_net,
        )
        actor = SoftmaxActor(actor_net, envs, args)

        optimizer = torch.optim.Adam(actor_net.parameters(), lr=args.lr)

        sampler = Sampler(actor=actor, env=envs, args=args)
        agent = PPO(actor, optimizer, args)

        writer = tb.SummaryWriter(logdir="ppo")
        step = 0
        for i in range(10):
            with tqdm(total=int(args.num_episodes / 10), desc="Iteration %d" % i) as pbar:
                for i_episode in range(int(args.num_episodes / 10)):
                    sampler.sample_episode(n_episode=1)
                    sample_data = sampler.get_sampler_data()
                    enhance_sample_data = enhance_advantage(sample_data)
                    torch.set_grad_enabled(True)
                    agent.updata_parameter(enhance_sample_data)
                    torch.set_grad_enabled(False)

                    PipeLearner.save_logging(writer, agent.logging, step)
                    PipeLearner.save_logging(writer, sampler.logging, step)
                    step += 1
                    pbar.update(1)


def ppo_args():
    parser = argparse.ArgumentParser()
    # env
    parser.add_argument("--task", type=str, default="CartPole-v0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env_num", type=int, default=1)
    # agent
    parser.add_argument("--num_episodes", type=int, default=500)
    parser.add_argument('--actor_net', type=list, default=[128])
    parser.add_argument("--epsilon", type=float, default=0.01)
    # learn
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--start_learn_buffer_size", type=int, default=500)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--update_target_interval", type=int, default=10)
    parser.add_argument("--device", type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_known_args()[0]
    return args


if __name__ == "__main__":
    ppo_args = ppo_args()
    PipeLearner.run(ppo_args)

