import torch
import argparse
import numpy as np

from XuanJing.utils.net.common import CarlaNet
from XuanJing.actor.actor_group.noise_actor import NoiseActor
from XuanJing.gamecore.fake.fake_carla_env import CarlaEnv
from XuanJing.learner.sample_step import PipeLearner
from XuanJing.algorithms.modelfree.ddpg import DDPG
from XuanJing.env.build_env import instance_env_vector


def carla_args():
    parser = argparse.ArgumentParser()
    # env
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env_num", type=int, default=1)
    # agent
    parser.add_argument("--alg", type=str, default="carla")
    parser.add_argument("--num_episodes", type=int, default=200)
    parser.add_argument('--actor_net', type=list, default=[64])
    # learn
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--start_learn_buffer_size", type=int, default=1000)
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-3)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--update_target_interval", type=int, default=10)
    parser.add_argument("--device", type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_known_args()[0]
    return args


def build_train(args):
    env = instance_env_vector(CarlaEnv())

    actor_net = CarlaNet(env)

    actor = NoiseActor(actor_net, env, args)
    optimizer = torch.optim.Adam(actor_net.parameters(), lr=args.actor_lr)
    agent = DDPG(
        actor,
        optimizer,
        args
    )

    pipeline = PipeLearner()
    pipeline.run(
        args,
        env,
        actor,
        agent
    )


if __name__ == "__main__":
    carla_args = carla_args()
    build_train(carla_args)
