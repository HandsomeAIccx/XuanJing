import copy
import torch
import numpy as np
from XuanJing.utils.torch_utils import tensorify
from XuanJing.buffer.replaybuffer import ReplayBuffer
import torch.nn.functional as F


class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class SAC(object):
    def __init__(
            self,
            actor,
            optim,
            args
    ):
        self.actor_net = actor.actor_net
        self.critic_1 = QValueNet(state_dim=4, hidden_dim=128)
        self.critic_2 = QValueNet(state_dim=4, hidden_dim=128)
        self.target_critic_1 = copy.deepcopy(self.critic_1)
        self.target_critic_2 = copy.deepcopy(self.critic_2)

        self.replay_buffer = ReplayBuffer(capacity=args.buffer_size)

        self.actor_optim = optim
        self.critic1_optim = torch.optim.Adam(self.critic_1.parameters(), lr=1e-2)
        self.critic2_optim = torch.optim.Adam(self.critic_2.parameters(), lr=1e-2)

        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=1e-2)
        self.target_entropy = -1  # 目标熵的大小 -env.action_space.shape
        self.tau = 0.005
        self.args = args
        self.learn_step = 0
        self.logging = {}

    # 计算目标Q值,直接用策略网络的输出概率进行期望计算
    def calc_target(self, rewards, next_states, dones):
        next_probs = F.softmax(self.actor_net(next_states), dim=1)
        next_log_probs = torch.log(next_probs + 1e-8)
        entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)
        q1_value = self.target_critic_1(next_states)
        q2_value = self.target_critic_2(next_states)
        min_qvalue = torch.sum(next_probs * torch.min(q1_value, q2_value), dim=1, keepdim=True)
        next_value = min_qvalue + self.log_alpha.exp() * entropy
        td_target = rewards + self.args.gamma * next_value * (1 - dones)
        return td_target

    def updata_parameter(
            self,
            train_data
    ):
        self.replay_buffer.push(train_data)

        if len(self.replay_buffer) < self.args.start_learn_buffer_size:
            return

        batch_data = self.replay_buffer.random_pop(self.args.batch_size)
        obs = tensorify(batch_data.get_value("obs"))
        actions = tensorify(batch_data.get_value("output")["act"]).view(-1, 1)
        next_obs = tensorify(batch_data.get_value("next_obs"))
        rewards = tensorify(batch_data.get_value("reward")).view(-1, 1)
        done = tensorify(batch_data.get_value("done")).view(-1, 1)

        #
        # 更新两个Q网络
        td_target = self.calc_target(rewards, next_obs, done)
        critic_1_q_values = self.critic_1(obs).gather(1, actions)
        critic_1_loss = torch.mean(F.mse_loss(critic_1_q_values, td_target.detach()))
        critic_2_q_values = self.critic_2(obs).gather(1, actions)
        critic_2_loss = torch.mean(F.mse_loss(critic_2_q_values, td_target.detach()))
        self.critic1_optim.zero_grad()
        critic_1_loss.backward()
        self.critic1_optim.step()
        self.critic2_optim.zero_grad()
        critic_2_loss.backward()
        self.critic2_optim.step()

        # 更新策略网络
        probs = F.softmax(self.actor_net(obs), dim=-1)
        log_probs = torch.log(probs + 1e-8)
        # 直接根据概率计算熵
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)  #
        q1_value = self.critic_1(obs)
        q2_value = self.critic_2(obs)
        min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value), dim=1, keepdim=True)  # 直接根据概率计算期望
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # 更新alpha值
        alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

        # anything you want to recorder!
        self.logging.update({
            "Learn/losses": actor_loss.item()
        })

    def sync_weight(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1 - self.tau) + param.data * self.tau)

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)