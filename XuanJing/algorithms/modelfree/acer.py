import torch
from XuanJing.utils.torch_utils import tensorify
import torch.nn.functional as F


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ACER(object):
    def __init__(
            self,
            actor,
            optim,
            args
    ):
        """
        arxiv: Sample Efficient Actor-Critic with Experience Replay
        """
        self.actor_net = actor.actor_net
        self.critic_net = ValueNet(state_dim=4, hidden_dim=128)
        self.actor_optim = optim
        self.critic_optim = torch.optim.Adam(self.critic_net.parameters(), lr=1e-2)
        self.args = args
        self.learn_step = 0
        self.logging = {}

    def compute_advantage(self, gamma, lmbda, td_delta):
        td_delta = td_delta.detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = gamma * lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.tensor(advantage_list, dtype=torch.float)

    def updata_parameter(
            self,
            train_data
    ):
        obs = tensorify(train_data.get_value("obs"))
        actions = tensorify(train_data.get_value("output")["act"]).view(-1, 1)
        next_obs = tensorify(train_data.get_value("next_obs"))
        rewards = tensorify(train_data.get_value("reward")).view(-1, 1)
        done = tensorify(train_data.get_value("done")).view(-1, 1)

        # TD
        td_target = rewards + self.args.gamma * self.critic_net(next_obs) * (1 - done)
        td_delta = td_target - self.critic_net(obs)

        advantage = self.compute_advantage(self.args.gamma, 0.95, td_delta)
        old_log_probs = torch.log(F.softmax(self.actor_net(obs), dim=1).gather(1, actions)).detach()

        for i in range(10):
            predict_prob = self.actor_net(obs)
            softmax_probs = F.softmax(predict_prob, dim=1)
            log_prob = torch.log(softmax_probs).gather(1, actions)

            ratio = torch.exp(log_prob - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * advantage

            actor_loss = torch.mean(- torch.min(surr1, surr2))

            critic_loss = torch.mean(F.mse_loss(self.critic_net(obs), td_target.detach()))

            self.actor_optim.zero_grad()
            self.critic_optim.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optim.step()
            self.critic_optim.step()

        # anything you want to recorder!
        self.logging.update({
            "Learn/losses": actor_loss.item()
        })