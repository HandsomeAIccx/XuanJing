import torch
import argparse
import numpy as np

from XuanJing.env.build_env import env_vector
from XuanJing.utils.net.common import MLP
from XuanJing.algorithms.modelfree.trpo import TRPO
from XuanJing.actor.actor_group.softmax_actor import SoftmaxActor
from XuanJing.learner.sample_episode import PipeLearner


def trpo_args():
    parser = argparse.ArgumentParser()
    # env
    parser.add_argument("--task", type=str, default="CartPole-v0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env_num", type=int, default=1)
    # agent
    parser.add_argument("--num_episodes", type=int, default=5000)
    parser.add_argument('--actor_net', type=list, default=[128])
    # learn
    parser.add_argument("--alg", type=str, default="trpo")
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--update_times", type=int, default=10)
    parser.add_argument("--start_learn_buffer_size", type=int, default=500)
    parser.add_argument("--actor_lr", type=float, default=2e-3)
    parser.add_argument("--critic_lr", type=float, default=1e-2)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--lmbda", type=float, default=0.95)
    parser.add_argument("--kl_constraint", type=float, default=0.0005)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--update_target_interval", type=int, default=10)
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_known_args()[0]
    return args


def build_train(args):
    env = env_vector(args=args)

    actor_net = MLP(
        input_dim=int(np.prod(env.observation_space.shape)),
        output_dim=int(np.prod(env.action_space.n)),
        hidden_sizes=args.actor_net
    )

    actor = SoftmaxActor(
        actor_net,
        env,
        args
    )

    optimizer = torch.optim.Adam(actor_net.parameters(), lr=args.actor_lr)

    agent = TRPO(actor, optimizer, args)

    pipeline = PipeLearner()
    pipeline.run(
        args,
        env,
        actor,
        agent
    )


if __name__ == "__main__":
    trpo_args = trpo_args()
    build_train(trpo_args)

