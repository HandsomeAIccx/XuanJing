import copy
import torch
import torch.nn.functional as F
from XuanJing.buffer.replaybuffer import ReplayBuffer
from XuanJing.utils.torch_utils import tensorify


class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)

class DDPG(object):
    def __init__(
            self,
            actor,
            optim,
            args
    ):
        super(DDPG, self).__init__()
        self.actor_net = actor.actor_net
        self.target_actor_net = copy.deepcopy(self.actor_net)
        self.critic_net = QValueNet(3, 128, 1).to(args.device)
        self.target_critic_net = copy.deepcopy(self.critic_net)
        self.target_critic_net.load_state_dict(self.critic_net.state_dict())
        self.target_actor_net.load_state_dict(self.actor_net.state_dict())
        self.actor_optimizer = optim
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=args.critic_lr)
        self.args = args
        self.replay_buffer = ReplayBuffer(capacity=args.buffer_size)
        self.learn_step = 0
        self.logging = {}

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
        reward = tensorify(batch_data.get_value("reward")).view(-1, 1)
        done = tensorify(batch_data.get_value("done")).view(-1, 1)

        next_q_values = self.target_critic_net(next_obs, self.target_actor_net(next_obs))
        q_targets = reward + self.args.gamma * next_q_values * (1 - done)
        critic_loss = torch.mean(F.mse_loss(self.critic_net(obs, actions), q_targets))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -torch.mean(self.critic_net(obs, self.actor_net(obs)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # if self.learn_step % self.args.update_target_interval == 0:
        self.soft_update(self.actor_net, self.target_actor_net)
        self.soft_update(self.critic_net, self.target_critic_net)
        self.learn_step += 1
        # anything you want to recorder!
        self.logging.update({
            "Learn/losses": actor_loss.item()
        })

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.args.tau) + param.data * self.args.tau)
