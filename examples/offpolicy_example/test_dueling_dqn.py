import torch
import argparse
import numpy as np

from XuanJing.utils.net.common import VAnet
from XuanJing.env.build_env import env_vector
from XuanJing.learner.sample_episode import PipeLearner
from XuanJing.algorithms.modelfree.dueling_dqn import DuelingDQN
from XuanJing.actor.actor_group.epsilon_greedy_actor import EpsGreedyActor


def dueling_dqn_args():
    parser = argparse.ArgumentParser()
    # env
    parser.add_argument("--task", type=str, default="CartPole-v0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env_num", type=int, default=1)
    # agent
    parser.add_argument("--alg", type=str, default="dueling_dqn")
    parser.add_argument("--num_episodes", type=int, default=2000)
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


def build_train(args):
    env = env_vector(args=args)

    actor_net = VAnet(
        input_dim=int(np.prod(env.observation_space.shape)),
        hidden_dim=args.actor_net[0],
        output_dim=int(np.prod(env.action_space.n))
    )

    actor = EpsGreedyActor(actor_net, env, args)
    optimizer = torch.optim.Adam(actor_net.parameters(), lr=args.lr)
    agent = DuelingDQN(actor, optimizer, args)

    pipeline = PipeLearner()
    pipeline.run(
        args,
        env,
        actor,
        agent
    )


if __name__ == "__main__":
    dueling_dqn_args = dueling_dqn_args()
    build_train(dueling_dqn_args)

