import torch
import random
import numpy as np

class BaseLearner(object):
    @staticmethod
    def save_logging(writer, log_data, step):
        assert isinstance(log_data, dict), f"input type {type(log_data)} is not a dict!"
        for k, v in log_data.items():
            writer.add_scalar(k, v, step)

    @staticmethod
    def set_global_seeds(seed):
        r""" Set the random seed.

        It sets the following dependencies with the given random seed:

        Args:
            seed (int): a given seed.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
