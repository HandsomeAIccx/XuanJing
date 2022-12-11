import torch
import torch.nn.functional as F

from XuanJing.utils.torch_utils import tensorify


class BehaviorClone(object):
    def __init__(self, actor, optim, args):
        self.args = args
        self.actor = actor
        self.optimizer = optim
        self.actor_net = actor.actor_net

    def update_parameter(self, states, actions):
        states = tensorify(states)
        actions = tensorify(actions).view(-1, 1).to(self.args.device)
        log_probs = torch.log(F.softmax(self.actor_net(states), dim=1).gather(1, actions))
        bc_loss = -log_probs.mean()  # 最大似然估计
        self.optimizer.zero_grad()
        bc_loss.backward()
        self.optimizer.step()

