import torch
import argparse
import numpy as np

from XuanJing.learner.sample_episode import PipeLearner
from XuanJing.algorithms.modelfree.ppo import PPO
from XuanJing.env.build_env import env_vector


def ppo_args():
    parser = argparse.ArgumentParser()
    # env
    parser.add_argument("--task", type=str, default="LunarLanderContinuous-v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env_num", type=int, default=1)
    # agent
    parser.add_argument("--num_episodes", type=int, default=500)
    parser.add_argument('--actor_net', type=list, default=[128])
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--ppo_clip", type=float, default=0.2)
    parser.add_argument("--kl_coeff", type=float, default=0.0)
    parser.add_argument("--entropy_coeff", type=float, default=0.01)
    # learn
    parser.add_argument("--alg", type=str, default="ppo")
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--update_times", type=int, default=10)
    parser.add_argument("--start_learn_buffer_size", type=int, default=500)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--update_target_interval", type=int, default=10)
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_known_args()[0]
    return args


def build_train(args):
    env = env_vector(args=args)

    actor = DiscreteActor(
        input_dim=int(np.prod(env.observation_space.shape)),
        output_dim=int(np.prod(env.action_space.n)),
        hidden_sizes=args.actor_net,
        activation=torch.nn.ReLU
    )

    optimizer = torch.optim.Adam(actor.parameters(), lr=args.lr)

    agent = PPO(actor, optimizer, args)

    pipeline = PipeLearner()
    pipeline.run(
        args,
        env,
        actor,
        agent
    )


if __name__ == "__main__":
    ppo_args = ppo_args()
    build_train(ppo_args)

