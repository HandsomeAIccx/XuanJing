import torch.optim
import numpy as np
from tqdm import tqdm


from XuanJing.env.sample.patch import Patch
from XuanJing.learner.base import BaseLearner
from XuanJing.env.sample.sampler import Sampler


class SelfplaySampler(Sampler):
    def __init__(
            self,
            actor,
            env,
            args
    ):
        """
        在IPPO中只有采样和更新训练过程与PPO不一样，因此这里将需要更改的两个模块单独拿出来，方便与PPO区分开。
        """
        super(SelfplaySampler, self).__init__(actor, env, args)

    def sample_self_play_episode(self, n_episode=0):
        assert n_episode > 0, "episode len must > 0!"

        episodes_patch_data_1 = []
        episodes_patch_data_2 = []
        for i in range(n_episode):
            episode_patch_data_1 = Patch()
            episode_patch_data_2 = Patch()
            obs, done = self.env.reset(), False
            terminal = False
            while not terminal:
                actor_out_1 = self.actor.sample_forward(obs[:, 0, :])
                actor_out_2 = self.actor.sample_forward(obs[:, 1, :])
                action = np.concatenate([actor_out_1['act'], actor_out_2['act']], axis=1)
                obs_next, reward, done, info = self.env.step(action)

                episode_patch_data_1.add(
                    Patch(
                        obs=obs[:, 0, :],
                        output=actor_out_1,
                        reward=reward[:, 0] + 100 if info[0]['win'] else reward[:, 0] - 0.1,
                        done=np.array([False]),
                        next_obs=obs_next[:, 0, :]
                    )
                )

                episode_patch_data_2.add(
                    Patch(
                        obs=obs[:, 1, :],
                        output=actor_out_2,
                        reward=reward[:, 1] + 100 if info[0]['win'] else reward[:, 1] - 0.1,
                        done=np.array([False]),
                        next_obs=obs_next[:, 1, :]
                    )
                )
                terminal = all(done[0])
                if all(done[0]):
                    episodes_patch_data_1.append(episode_patch_data_1)
                    episodes_patch_data_2.append(episode_patch_data_2)
                obs = obs_next

        return episodes_patch_data_1, episodes_patch_data_2, 1 if info[0]["win"] else 0


class PipeLearner(BaseLearner):
    @staticmethod
    def run(
            args,
            env,
            actor,
            agent
    ):
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        sampler = SelfplaySampler(actor=actor, env=env, args=args)
        win_list = []
        tqdm_range = tqdm(range(args.num_episodes))
        for i_episode in tqdm_range:
            sample_data_1, sample_data_2, win = sampler.sample_self_play_episode(n_episode=1)
            win_list.append(win)
            torch.set_grad_enabled(True)
            agent.updata_parameter(sample_data_1[0])
            agent.updata_parameter(sample_data_2[0])
            torch.set_grad_enabled(False)

            tqdm_range.set_postfix({"Episode": f"{i_episode}", "AverageEpisodeReward": f"{np.mean(win_list[-100:])}"})
