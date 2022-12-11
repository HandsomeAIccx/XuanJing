import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


from XuanJing.utils.net.common import MLP
from XuanJing.env.sample.patch import Patch
from XuanJing.env.build_env import env_vector
from XuanJing.env.sample.sampler import Sampler
from XuanJing.algorithms.modelfree.ppo import PPO
from XuanJing.learner.sample_episode import PipeLearner
from XuanJing.algorithms.imitation.bc import BehaviorClone
from XuanJing.actor.actor_group.softmax_actor import SoftmaxActor


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
    parser.add_argument("--alg", type=str, default="ppo")
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--update_times", type=int, default=10)
    parser.add_argument("--start_learn_buffer_size", type=int, default=500)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--update_target_interval", type=int, default=10)
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_known_args()[0]
    return args


def build_train_ppo(args):
    env = env_vector(args=args)

    actor_net = MLP(
        input_dim=int(np.prod(env.observation_space.shape)),
        output_dim=int(np.prod(env.action_space.n)),
        hidden_sizes=args.actor_net,
    )
    actor = SoftmaxActor(actor_net, env, args)

    optimizer = torch.optim.Adam(actor_net.parameters(), lr=args.lr)

    agent = PPO(actor, optimizer, args)

    pipeline = PipeLearner()
    pipeline.run(
        args,
        env,
        actor,
        agent
    )
    return agent


def sample_expert_data(n_episode, agent, env, args):
    actor = agent.actor
    sampler = Sampler(actor=actor, env=env, args=args)
    patch_episodes = sampler.sample_episode(n_episode=n_episode)
    post_data = Patch()
    for patch in patch_episodes:
        post_data.add(patch)
    return post_data


def bc_args():
    parser = argparse.ArgumentParser()
    # bc
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_iterations", type=int, default=1000)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_known_args()[0]
    return args


def build_train_bc(args):

    bc_actor_net = MLP(
        input_dim=int(np.prod(env.observation_space.shape)),
        output_dim=int(np.prod(env.action_space.n)),
        hidden_sizes=[128],
    )
    actor = SoftmaxActor(bc_actor_net, env, args)
    optimizer = torch.optim.Adam(bc_actor_net.parameters(), lr=args.lr)
    bc_agent = BehaviorClone(
        actor,
        optimizer,
        args,
    )

    post_data = sample_expert_data(
        n_episode=1,
        agent=ppo_agent,
        env=env,
        args=ppo_args
    )

    expert_state = post_data.get_value('obs')
    expert_action = post_data.get_value('output')['act']

    random_index = random.sample(range(expert_state.shape[0]), bc_args.n_samples)
    expert_s = expert_state[random_index]
    expert_a = expert_action[random_index]
    test_returns = []
    with tqdm(total=bc_args.n_iterations, desc="进度条") as pbar:
        for i in range(bc_args.n_iterations):
            sample_indices = np.random.randint(low=0, high=expert_s.shape[0], size=bc_args.batch_size)
            torch.set_grad_enabled(True)
            bc_agent.update_parameter(expert_s[sample_indices], expert_a[sample_indices])
            torch.set_grad_enabled(False)
            post_data = sample_expert_data(
                n_episode=5,
                agent=bc_agent,
                env=env,
                args=bc_args
            )

            current_return = np.sum(post_data.get_value('reward')) / 5
            test_returns.append(current_return)
            if (i + 1) % 10 == 0:
                pbar.set_postfix({'return': '%.3f' % np.mean(test_returns[-10:])})
            pbar.update(1)

    iteration_list = list(range(len(test_returns)))
    plt.plot(iteration_list, test_returns)
    plt.xlabel('Iterations')
    plt.ylabel('Returns')
    plt.title('BC on {}'.format(ppo_args.task))
    plt.show()


if __name__ == "__main__":
    # obtain expert data.
    ppo_args = ppo_args()
    ppo_agent = build_train_ppo(ppo_args)
    env = env_vector(args=ppo_args)

    # bc process
    bc_args = bc_args()
    build_train_bc(bc_args)
