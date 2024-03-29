import copy

from XuanJing.env.sample.patch import Patch
from XuanJing.utils.torch_utils import tensorify


class Sampler(object):
    def __init__(
            self,
            actor,
            env,
            args
    ):
        super(Sampler, self).__init__()
        self.actor = actor
        self.env = env
        self.args = args
        self.patch_data = Patch()
        self.init_state = True
        self.current_obs = None
        self.episode_reward = 0
        self.episodes_reward = []
        self.episode_done = False
        self.logging = {}

    def sample_episode(self, n_episode=0):
        assert n_episode > 0, "episode len must > 0!"

        episodes_patch_data = []
        for i in range(n_episode):
            episode_patch_data = Patch()
            obs, done = self.env.reset(), False
            while not done:
                actor_out = self.actor.sample_action(tensorify(obs))
                action = actor_out['act']
                obs_next, reward, done, info = self.env.step(action)
                episode_patch_data.add(
                    Patch(
                        obs=obs,
                        output=actor_out,
                        reward=reward,
                        done=done,
                        next_obs=obs_next
                    )
                )
                if done:
                    episodes_patch_data.append(episode_patch_data)
                obs = obs_next

        return episodes_patch_data

    def sample_step(self, n_step=0):
        assert n_step > 0, "n_step len must > 0!"
        cur_step = 0
        if self.current_obs is None:
            self.current_obs = self.env.reset()
        while True:
            actor_out = self.actor.sample_forward(self.current_obs)
            obs_next, reward, done, info = self.env.step(actor_out['act'])
            self.episode_reward += reward[0]
            self.patch_data.add(
                Patch(
                    obs=self.current_obs,
                    output=actor_out,
                    reward=reward,
                    done=done,
                    next_obs=obs_next
                )
            )
            if done:
                obs_next = self.env.reset()
                self.episodes_reward.append(self.episode_reward)
                self.episode_reward = 0
                self.episode_done = True
            else:
                self.episode_done = False

            self.current_obs = obs_next

            cur_step += 1
            if cur_step >= n_step:
                break

    def get_episode_result(self):
        return {"episodes_reward": self.episodes_reward}

    def get_sampler_data(self):
        patch_data = copy.copy(self.patch_data)
        self.patch_data.clear()
        return patch_data

    def replace_actor(self, actor):
        self.actor.actor_net = actor