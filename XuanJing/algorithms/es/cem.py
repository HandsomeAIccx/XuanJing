import torch


class CemAgent(object):
    """
    "Implementation of "Cross Entropy Method."
    ref: https://github.com/tinyzqh/Cross-entropy-method-Series/blob/main/cem.py.
    """
    def __init__(self,
                 actor_net,
                 args):
        self.actor_net = actor_net
        self.sigma = args.sigma
        self.lr = args.learning_rate
        self.population_size = args.population_size
        self.noise = None
        self.n_elite = int(args.population_size * args.elite_frac)

    def generate_population(self, population_size):
        parameters = self.actor_net.to_vec()
        self.noise = torch.randn(population_size, parameters.shape[0])
        springs = self.noise * self.sigma
        return springs + parameters, self.noise

    def update_actor_para(self, param):
        self.actor_net.from_vec(param)

    def update_net(self, population_reward, population):
        elite_ids = population_reward.argsort()[-self.n_elite:]
        param_mean = population[elite_ids].mean(dim=0)
        self.actor_net.from_vec(param_mean)
