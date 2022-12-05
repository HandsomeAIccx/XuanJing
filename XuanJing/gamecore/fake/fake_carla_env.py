import gym
from gym import spaces
import numpy as np
from gym.utils import seeding
import copy


class RelativePosition(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._observation_space = spaces.Box(np.array([-2, -1, -5, 0]), np.array([2, 1, 30, 1]), dtype=np.float32)

    def observation(self, obs):
        return obs['state']


class CarlaEnv(gym.Env):
    def __init__(self):
        self.obs_size = 256
        observation_space_dict = {
            'camera': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
            'lidar': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
            'birdeye': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
            'state': spaces.Box(np.array([-2, -1, -5, 0]), np.array([2, 1, 30, 1]), dtype=np.float32)
        }

        self.observation_space = spaces.Dict(observation_space_dict)
        self.action_space = spaces.Box(np.array([0, 0]),
                                       np.array([1, 1]),
                                       dtype=np.float32)  # acc, steer

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        return self._get_obs()

    def step(self, action):
        info = {'waypoints': [0], 'vehicle_front': [0]}
        return self._get_obs(), self._get_reward(), self._terminal(), copy.deepcopy(info)

    def _get_reward(self):
        return np.array([1])

    def _terminal(self):
        a = np.random.uniform(0.0,1.0)
        if a < 0.96:
            return np.array([False])
        else:
            return np.array([True])

    def _get_obs(self):
        camera = np.random.rand(256, 256, 3)
        lidar = np.random.rand(256, 256, 3)
        birdeye = np.random.rand(256,256,3)
        state = np.random.rand(4)
        obs = {
            'camera': camera.astype(np.uint8),
            'lidar': lidar.astype(np.uint8),
            'birdeye': birdeye.astype(np.uint8),
            'state': state,
        }
        return obs


if __name__ == "__main__":
    env_instance = CarlaEnv()
    # env = RelativePosition(Carla_env())
    # env = gym.wrappers.FrameStack(env,4)
    obs = env_instance.reset()

    for i in range(1000):
        done = False
        while not done:
            action = 1
            next_obs, reward, done, info = env_instance.step(action)
            print(obs)

