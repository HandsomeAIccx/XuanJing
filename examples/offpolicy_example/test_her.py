import torch
import numpy as np
import argparse

import random
from tqdm import tqdm
import collections
import matplotlib.pyplot as plt

from XuanJing.algorithms.modelfree.her import DDPG
from XuanJing.gamecore.classic.world_env import WorldEnv


class Trajectory:
    ''' 用来记录一条完整轨迹 '''
    def __init__(self, init_state):
        self.states = [init_state]
        self.actions = []
        self.rewards = []
        self.dones = []
        self.length = 0

    def store_step(self, action, state, reward, done):
        self.actions.append(action)
        self.states.append(state)
        self.rewards.append(reward)
        self.dones.append(done)
        self.length += 1


class ReplayBuffer_Trajectory:
    ''' 存储轨迹的经验回放池 '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add_trajectory(self, trajectory):
        self.buffer.append(trajectory)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size, use_her, dis_threshold=0.15, her_ratio=0.8):
        batch = dict(states=[],
                     actions=[],
                     next_states=[],
                     rewards=[],
                     dones=[])
        for _ in range(batch_size):
            traj = random.sample(self.buffer, 1)[0]
            step_state = np.random.randint(traj.length)
            state = traj.states[step_state]
            next_state = traj.states[step_state + 1]
            action = traj.actions[step_state]
            reward = traj.rewards[step_state]
            done = traj.dones[step_state]

            if use_her and np.random.uniform() <= her_ratio:
                step_goal = np.random.randint(step_state + 1, traj.length + 1)
                goal = traj.states[step_goal][:2]  # 使用HER算法的future方案设置目标
                dis = np.sqrt(np.sum(np.square(next_state[:2] - goal)))
                reward = -1.0 if dis > dis_threshold else 0
                done = False if dis > dis_threshold else True
                state = np.hstack((state[:2], goal))
                next_state = np.hstack((next_state[:2], goal))

            batch['states'].append(state)
            batch['next_states'].append(next_state)
            batch['actions'].append(action)
            batch['rewards'].append(reward)
            batch['dones'].append(done)

        batch['states'] = np.array(batch['states'])
        batch['next_states'] = np.array(batch['next_states'])
        batch['actions'] = np.array(batch['actions'])
        return batch


def her_args():
    parser = argparse.ArgumentParser()
    # env
    parser.add_argument("--task", type=str, default='CartPole-v0')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env_num", type=int, default=1)
    # agent
    parser.add_argument("--num_episodes", type=int, default=2000)
    parser.add_argument("--actor_lr", type=float, default=1e-3)
    parser.add_argument("--critic_lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--state_dim", type=int, default=4)
    parser.add_argument("--action_dim", type=int, default=2)
    parser.add_argument("--action_bound", type=int, default=1)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--n_train", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--minimal_episodes", type=int, default=200)
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_known_args()[0]
    return args


def build_train(args):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    env = WorldEnv()
    replay_buffer = ReplayBuffer_Trajectory(args.buffer_size)
    agent = DDPG(args)

    return_list = []
    for i in range(10):
        with tqdm(total=int(args.num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(args.num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                traj = Trajectory(state)
                done = False
                while not done:
                    action = agent.take_action(state)
                    state, reward, done = env.step(action)
                    episode_return += reward
                    traj.store_step(action, state, reward, done)
                replay_buffer.add_trajectory(traj)
                return_list.append(episode_return)
                if replay_buffer.size() >= args.minimal_episodes:
                    for _ in range(args.n_train):
                        transition_dict = replay_buffer.sample(args.batch_size, True)
                        agent.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                            '%d' % (args.num_episodes / 10 * i + i_episode + 1),
                        'return':
                            '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DDPG with HER on {}'.format('GridWorld'))
    plt.show()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    env = WorldEnv()
    replay_buffer = ReplayBuffer_Trajectory(args.buffer_size)
    agent = DDPG(args)

    return_list = []
    for i in range(10):
        with tqdm(total=int(args.num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(args.num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                traj = Trajectory(state)
                done = False
                while not done:
                    action = agent.take_action(state)
                    state, reward, done = env.step(action)
                    episode_return += reward
                    traj.store_step(action, state, reward, done)
                replay_buffer.add_trajectory(traj)
                return_list.append(episode_return)
                if replay_buffer.size() >= args.minimal_episodes:
                    for _ in range(args.n_train):
                        # 和使用HER训练的唯一区别
                        transition_dict = replay_buffer.sample(args.batch_size, False)
                        agent.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                            '%d' % (args.num_episodes / 10 * i + i_episode + 1),
                        'return':
                            '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DDPG without HER on {}'.format('GridWorld'))
    plt.show()


if __name__ == "__main__":

    her_args = her_args()
    build_train(her_args)