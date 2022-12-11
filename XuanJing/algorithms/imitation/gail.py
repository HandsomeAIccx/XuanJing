import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()


class GAIL(object):
    def __init__(self, actor, optim, args):
        self.actor_net = actor.actor_net
        self.discriminator = None
        self.optim = None

    def updata_parameter(
            self,
            train_data
    ):
        pass