import torch
import copy
from torch import Tensor
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions import kl_divergence

from XuanJing.utils.torch_utils import tensorify
from XuanJing.utils.net.common import linear_lr_scheduler


class ConnectionValueNet(torch.nn.Module):
    def __init__(self, actor):
        super(ConnectionValueNet, self).__init__()
        self.base_value_net = copy.deepcopy(actor)
        self.base_value_net_list = list(self.base_value_net.modules())
        self.last_layer_units = self.base_value_net_list[-1].out_features
        self.base_value_net.add_module("last_value_layer", nn.Linear(self.last_layer_units, 1))

    def forward(self, x):
        return self.base_value_net(x)


class PPO(object):
    def __init__(
            self,
            agent_model,
            args
    ):
        self.actor = agent_model.models['actor']
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic = agent_model.models['critic']
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)
        self.args = args
        self.learn_step = 0
        self.criterion = torch.nn.SmoothL1Loss()
        self.policy_lr_scheduler = linear_lr_scheduler(
            self.actor_optim,
            args.scheduling_lr_bound,
            args.scheduling_min_lr
        )
        self.logging = {}

    def compute_advantage(self, gamma, lmbda, td_delta):
        td_delta = td_delta.detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = gamma * lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.tensor(np.array(advantage_list), dtype=torch.float)

    def updata_parameter(
            self,
            train_data
    ):
        obs = tensorify(train_data.get_value("obs"))
        action_b = tensorify(train_data.get_value("output")["act"])
        action_logit_b = tensorify(train_data.get_value("output")["logit"])
        next_obs = tensorify(train_data.get_value("next_obs"))
        rewards = tensorify(train_data.get_value("reward")).view(-1, 1)
        done = tensorify(train_data.get_value("done")).view(-1, 1)

        output_logit = self.actor(obs)

        # Step 1: value loss
        td_target = rewards + self.args.gamma * self.critic(next_obs) * (1 - done)
        current_state_value = self.critic(obs)
        td_delta = td_target - self.critic(obs)

        critic_loss = F.mse_loss(current_state_value, td_target.detach()).mean()

        advantage = self.compute_advantage(self.args.gamma, self.args.gae_lambda, td_delta).squeeze(1)

        # Step 2: policy loss
        kls = {}
        entropys = {}
        log_ratio = 0

        dist = Categorical(logits=output_logit)
        dist_b = Categorical(logits=action_logit_b)

        log_ratio = log_ratio + (dist.log_prob(action_b.squeeze(1)) - dist_b.log_prob(action_b.squeeze(1)))
        kls['action'] = kl_divergence(dist_b, dist).mean()
        entropys['action'] = dist.entropy().mean()

        kl = sum(kls.values())
        entropy = sum(entropys.values())
        ratio = log_ratio.exp()

        policy_loss = -torch.min(
            ratio * advantage,
            ratio.clip(1-self.args.ppo_clip, 1+self.args.ppo_clip) * advantage
        ).mean()

        policy_loss = (
            policy_loss
            + kl
            + entropy
        )

        upper_clip_rate = ratio.gt(1 + self.args.ppo_clip).float().mean()
        lower_clip_rate = ratio.lt(1 - self.args.ppo_clip).float().mean()

        self.optimizer_update(self.actor_optim, policy_loss)
        self.policy_lr_scheduler.step(self.learn_step)
        self.optimizer_update(self.critic_optim, critic_loss)

        # anything you want to recorder!
        self.logging.update({
            "Learn/policy_loss": policy_loss / self.args.update_times,
            "Learn/critic_loss": critic_loss / self.args.update_times,
            "Learn/upper_clip_rate": upper_clip_rate,
            "Learn/lower_clip_rate": lower_clip_rate
        })

    @staticmethod
    def optimizer_update(optimizer, loss: Tensor):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()