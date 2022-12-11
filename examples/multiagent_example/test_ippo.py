import torch
import argparse

from XuanJing.utils.net.common import MLP
from XuanJing.env.build_env import instance_env_vector
from XuanJing.gamecore.ma_gym.envs.combat.combat import Combat
from XuanJing.actor.actor_group.softmax_actor import SoftmaxActor
from XuanJing.algorithms.modelfree.ppo import PPO
from XuanJing.algorithms.multiagent.ippo import PipeLearner


def ippo_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    # agent
    parser.add_argument("--num_episodes", type=int, default=100000)
    # learn
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lmbda", type=float, default=0.97)
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--update_times", type=int, default=1)
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_known_args()[0]
    return args


def build_train(args):
    env = Combat(grid_shape=(15, 15), n_agents=2, n_opponents=2)
    env = instance_env_vector(env)

    actor_net = MLP(
        input_dim=env.observation_space[0].shape[0],
        output_dim=env.action_space[0].n,
        hidden_sizes=[args.hidden_dim]
    )

    actor = SoftmaxActor(actor_net, env, args)

    optimizer = torch.optim.Adam(actor_net.parameters(), lr=args.actor_lr)

    agent = PPO(
        actor,
        optimizer,
        args,
    )

    pipeline = PipeLearner()
    pipeline.run(
        args,
        env,
        actor,
        agent
    )


if __name__ == "__main__":
    ippo_args = ippo_args()
    build_train(ippo_args)