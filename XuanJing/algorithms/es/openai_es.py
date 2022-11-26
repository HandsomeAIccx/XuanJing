import torch


class EsAgent(object):
    """
    "Implementation of "Evolution Strategies as a Scalable Alternative to Reinforcement Learning"
    """
    def __init__(self,
                 actor_net,
                 args):
        self.actor_net = actor_net
        self.sigma = args.sigma
        self.lr = args.learning_rate
        self.param_size = self.actor_net.to_vec().shape[0]
        self.population_size = args.population_size
        self.noise = None

    def generate_population(self, population_size):
        parameters = self.actor_net.to_vec()
        self.noise = torch.randn(population_size, parameters.shape[0])
        springs = self.noise * self.sigma
        return springs + parameters, self.noise

    def update_actor_para(self, param):
        self.actor_net.from_vec(param)

    def update_net(self, population_reward):
        adv = (population_reward - population_reward.mean()) / population_reward.std()
        parameters = self.actor_net.to_vec()
        parameters = parameters + self.lr / (self.population_size * self.sigma) * torch.matmul(self.noise.T, adv)
        self.actor_net.from_vec(parameters)
        self.lr *= 0.9992354
