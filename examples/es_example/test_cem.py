import argparse
import numpy as np
from XuanJing.utils.net.common import MLP
from XuanJing.env.build_env import env_vector
from XuanJing.actor.actor_group.es_actor import EsActor
from XuanJing.algorithms.es.cem import CemAgent
from XuanJing.learner.espolicy import PipeLearner


def cem_args():
    parser = argparse.ArgumentParser()
    # env
    parser.add_argument("--task", type=str, default="CartPole-v0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env_num", type=int, default=1)
    # agent
    parser.add_argument('--actor_net', type=list, default=[128])
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--n_sessions", type=int, default=50)
    parser.add_argument("--population_size", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--elite_frac", type=float, help="fraction of samples used as elite set", default=0.2)

    args = parser.parse_known_args()[0]
    return args


def build_train(args):

    env = env_vector(args=args)

    actor_net = MLP(
        input_dim=int(np.prod(env.observation_space.shape)),
        output_dim=int(np.prod(env.action_space.n)),
        hidden_sizes=args.actor_net,
    )
    actor = EsActor(actor_net, env, args)
    agent = CemAgent(
        actor_net,
        args
    )
    PipeLearner.run(
        args,
        env,
        actor,
        agent
    )


if __name__ == "__main__":
    cem_args = cem_args()
    build_train(cem_args)
