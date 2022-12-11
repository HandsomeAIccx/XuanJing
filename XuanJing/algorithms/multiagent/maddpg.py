import argparse

import torch
import random
import collections
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

import XuanJing.gamecore.multiagent.scenarios as scenarios
from XuanJing.gamecore.multiagent.environment import MultiAgentEnv


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)


def make_env(scenario_name):
    # 从环境文件脚本中创建环境
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


def onehot_from_logits(logits, eps=0.01):
    ''' 生成最优动作的独热（one-hot）形式 '''
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    # 生成随机动作,转换成独热形式
    rand_acs = torch.autograd.Variable(torch.eye(logits.shape[1])[[
        np.random.choice(range(logits.shape[1]), size=logits.shape[0])
    ]],
                                       requires_grad=False).to(logits.device)
    # 通过epsilon-贪婪算法来选择用哪个动作
    return torch.stack([
        argmax_acs[i] if r > eps else rand_acs[i]
        for i, r in enumerate(torch.rand(logits.shape[0]))
    ])


def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """从Gumbel(0,1)分布中采样"""
    U = torch.autograd.Variable(tens_type(*shape).uniform_(),
                                requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """ 从Gumbel-Softmax分布中采样"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data)).to(
        logits.device)
    return F.softmax(y / temperature, dim=1)


def gumbel_softmax(logits, temperature=1.0):
    """从Gumbel-Softmax分布中采样,并进行离散化"""
    y = gumbel_softmax_sample(logits, temperature)
    y_hard = onehot_from_logits(y)
    y = (y_hard.to(logits.device) - y).detach() + y
    # 返回一个y_hard的独热量,但是它的梯度是y,我们既能够得到一个与环境交互的离散动作,又可以
    # 正确地反传梯度
    return y


class TwoLayerFC(torch.nn.Module):
    def __init__(self, num_in, num_out, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(num_in, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, num_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DDPG:
    ''' DDPG算法 '''
    def __init__(self, state_dim, action_dim, args):
        self.actor = TwoLayerFC(state_dim, action_dim, args.hidden_dim).to(args.device)
        self.target_actor = TwoLayerFC(state_dim, action_dim, args.hidden_dim).to(args.device)
        self.critic = TwoLayerFC(args.critic_input_dim, 1, args.hidden_dim).to(args.device)
        self.target_critic = TwoLayerFC(args.critic_input_dim, 1, args.hidden_dim).to(args.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)

    def take_action(self, state, explore=False):
        action = self.actor(state)
        if explore:
            action = gumbel_softmax(action)
        else:
            action = onehot_from_logits(action)
        return action.detach().cpu().numpy()[0]

    def soft_update(self, net, target_net, tau):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)


class MADDPG:
    def __init__(self, env, args):
        self.agents = []
        self.args = args
        for i in range(args.agent_num):
            self.agents.append(
                DDPG(args.state_dims[i], args.action_dims[i], args))
        self.gamma = args.gamma
        self.tau = args.tau
        self.critic_criterion = torch.nn.MSELoss()
        self.device = args.device

    @property
    def policies(self):
        return [agt.actor for agt in self.agents]

    @property
    def target_policies(self):
        return [agt.target_actor for agt in self.agents]

    def take_action(self, states, explore):
        states = [
            torch.tensor([states[i]], dtype=torch.float, device=self.device)
            for i in range(self.args.agent_num)
        ]
        return [
            agent.take_action(state, explore)
            for agent, state in zip(self.agents, states)
        ]

    def update(self, sample, i_agent):
        obs, act, rew, next_obs, done = sample
        cur_agent = self.agents[i_agent]

        cur_agent.critic_optimizer.zero_grad()
        all_target_act = [
            onehot_from_logits(pi(_next_obs))
            for pi, _next_obs in zip(self.target_policies, next_obs)
        ]
        target_critic_input = torch.cat((*next_obs, *all_target_act), dim=1)
        target_critic_value = rew[i_agent].view(-1,
            1) + self.gamma * cur_agent.target_critic(target_critic_input) * (1 - done[i_agent].view(-1, 1))
        critic_input = torch.cat((*obs, *act), dim=1)
        critic_value = cur_agent.critic(critic_input)
        critic_loss = self.critic_criterion(critic_value, target_critic_value.detach())
        critic_loss.backward()
        cur_agent.critic_optimizer.step()

        cur_agent.actor_optimizer.zero_grad()
        cur_actor_out = cur_agent.actor(obs[i_agent])
        cur_act_vf_in = gumbel_softmax(cur_actor_out)
        all_actor_acs = []
        for i, (pi, _obs) in enumerate(zip(self.policies, obs)):
            if i == i_agent:
                all_actor_acs.append(cur_act_vf_in)
            else:
                all_actor_acs.append(onehot_from_logits(pi(_obs)))
        vf_in = torch.cat((*obs, *all_actor_acs), dim=1)
        actor_loss = -cur_agent.critic(vf_in).mean()
        actor_loss += (cur_actor_out**2).mean() * 1e-3
        actor_loss.backward()
        cur_agent.actor_optimizer.step()

    def update_all_targets(self):
        for agt in self.agents:
            agt.soft_update(agt.actor, agt.target_actor, self.tau)
            agt.soft_update(agt.critic, agt.target_critic, self.tau)


def maddpg_args():
    parser = argparse.ArgumentParser()
    # env
    parser.add_argument("--num_episodes", type=int, default=5000)
    parser.add_argument("--episode_length", type=int, default=25)
    parser.add_argument("--buffer_size", type=int, default=100000)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--actor_lr", type=float, default=1e-2)
    parser.add_argument("--critic_lr", type=float, default=1e-2)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--tau", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--update_interval", type=int, default=100)
    parser.add_argument("--minimal_size", type=int, default=4000)
    parser.add_argument("--agent_num", type=int, default=3)

    args = parser.parse_known_args()[0]
    return args


def build_train(args):
    env_id = "simple_adversary"
    env = make_env(env_id)
    replay_buffer = ReplayBuffer(args.buffer_size)

    state_dims = []
    action_dims = []
    for action_space in env.action_space:
        action_dims.append(action_space.n)
    for state_space in env.observation_space:
        state_dims.append(state_space.shape[0])
    critic_input_dim = sum(state_dims) + sum(action_dims)
    args.state_dims = state_dims
    args.action_dims = action_dims
    args.critic_input_dim = critic_input_dim
    maddpg = MADDPG(env, args)

    def evaluate(env_id, maddpg, n_episode=10, episode_length=25):
        # 对学习的策略进行评估,此时不会进行探索
        env = make_env(env_id)
        returns = np.zeros(len(env.agents))
        for _ in range(n_episode):
            obs = env.reset()
            for t_i in range(episode_length):
                actions = maddpg.take_action(obs, explore=False)
                obs, rew, done, info = env.step(actions)
                rew = np.array(rew)
                returns += rew / n_episode
        return returns.tolist()

    return_list = []  # 记录每一轮的回报（return）
    total_step = 0
    for i_episode in range(args.num_episodes):
        state = env.reset()
        for e_i in range(args.episode_length):
            actions = maddpg.take_action(state, explore=True)
            next_state, reward, done, _ = env.step(actions)
            replay_buffer.add(state, actions, reward, next_state, done)
            state = next_state

            total_step += 1
            if replay_buffer.size() >= args.minimal_size and total_step % args.update_interval == 0:
                sample = replay_buffer.sample(args.batch_size)

                def stack_array(x):
                    rearranged = [[sub_x[i] for sub_x in x] for i in range(len(x[0]))]
                    return [
                        torch.FloatTensor(np.vstack(aa)).to(args.device)
                        for aa in rearranged
                    ]

                sample = [stack_array(x) for x in sample]
                for a_i in range(len(env.agents)):
                    maddpg.update(sample, a_i)
                maddpg.update_all_targets()
        if (i_episode + 1) % 100 == 0:
            ep_returns = evaluate(env_id, maddpg, n_episode=100)
            return_list.append(ep_returns)
            print(f"Episode: {i_episode + 1}, {ep_returns}")

    return_array = np.array(return_list)
    for i, agent_name in enumerate(["adversary_0", "agent_0", "agent_1"]):
        plt.figure()
        plt.plot(
            np.arange(return_array.shape[0]) * 100,
            moving_average(return_array[:, i], 9))
        plt.xlabel("Episodes")
        plt.ylabel("Returns")
        plt.title(f"{agent_name} by MADDPG")
    plt.show()


if __name__ == "__main__":
    maddpg_args = maddpg_args()
    build_train(maddpg_args)

