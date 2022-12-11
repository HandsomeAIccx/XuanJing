import gym
import numpy as np
from abc import ABC, abstractmethod

from XuanJing.env.vector.utils import stack_dict
from XuanJing.gamecore.fake.fake_carla_env import CarlaEnv


class BaseVectorEnv(ABC):
    def __init__(self, env_fns, reset_after_done):
        self._env_fns = env_fns
        self.env_num = len(env_fns)
        self._reset_after_done = reset_after_done

    def is_reset_after_done(self):
        return self._reset_after_done

    def __len__(self):
        return self.env_num

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def seed(self, seed=None):
        pass

    @abstractmethod
    def render(self):
        pass

    @abstractmethod
    def close(self):
        pass


class VectorEnv(object):
    """
    Base class for vectorized environment wrapper.

    Args:
        env_fns: A list of environment functions to create the environments from

    Usage:
    ::
        num_envs = 6
        env_fns = [env_fn1, env_fn2, ...]
        envs = VectorEnv(env_fns)
    """

    def __init__(self, env_fns):
        """
        Initialize the vectorized environment wrapper.
        Args:
            env_fns (list): List of environment creation functions
        """

        assert len(env_fns) > 0, "Number of environments must be positive."
        self._env_fns = env_fns
        self._env_num = len(env_fns)
        self.envs = [env_fun() for env_fun in env_fns]

        self.observation_space = self.envs[0].observation_space
        self.obs_type = type(self.observation_space)

        self.action_space = self.envs[0].action_space
        self.action_type = type(self.action_space)

        if self.obs_type == gym.spaces.Dict:
            self.obs_space_key = list(self.envs[0].observation_space.spaces.keys())
        else:
            self.obs_space_key = None

    def reset(self):
        """
        Reset all environments and return the stacked observations.
        Returns:
            dict/numpy.ndarray: Stacked observations
        """
        obses = [e.reset() for e in self.envs]
        if self.obs_type == gym.spaces.Dict:
            return_res = stack_dict(obses, self.obs_space_key)
            for key in self.obs_space_key:
                assert return_res[key].shape[0] == len(self.envs), \
                    f"obs's shape[0] {return_res.shape[0]} != {len(self.envs)}"
            return return_res
        elif self.obs_type == gym.spaces.box.Box:
            return_res = np.stack(obses)
            assert len(return_res.shape) == 2, f"obs's shape dim is {len(return_res.shape)}, required 2 dims."
            assert return_res.shape[0] == len(self.envs), f"obs's shape[0] {return_res.shape[0]} != {len(self.envs)}"
            return return_res
        else:
            raise ValueError("Not Implement Yet!")

    def step(self, action):
        assert len(action) == len(self.envs), "Env Num Error!"
        assert len(action.shape) == 2, "Length of given action shape must be 2 dims."
        if self.action_type == gym.spaces.discrete.Discrete:
            result = zip(*[e.step(a[0]) for e, a in zip(self.envs, action)])
        elif self.action_type == gym.spaces.box.Box:
            result = zip(*[e.step(a) for e, a in zip(self.envs, action)])
        else:
            raise ValueError(f"Not Implement action_type {self.action_type}")
        obses, reward, done, infos = result
        if self.obs_type == gym.spaces.Dict:
            return_obs = stack_dict(obses, self.obs_space_key)
            return_reward = np.stack(reward)
            return_done = np.stack(done)
            return_info = stack_dict(infos, list(infos[0].keys()))
            return return_obs, return_reward, return_done, return_info
        return map(np.stack, [obses, reward, done, infos])

    def seed(self):
        return [self._env_fns[id].seed() for id in range(self._env_num)]

    def render(self):
        pass

    def close(self):
        pass


if __name__ == "__main__":
    env_num = 6
    # envs = VectorEnv([lambda: gym.make('CartPole-v1') for _ in range(env_num)])
    envs = VectorEnv([lambda: CarlaEnv() for _ in range(env_num)])
    obs = envs.reset()
    obs, rew, done, info = envs.step([1] * env_num)
    envs.close()
