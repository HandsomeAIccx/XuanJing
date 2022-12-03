import os
import torch
import torch.optim
import numpy as np
from tqdm import tqdm
from pathlib import Path
import tensorboardX as tb

from XuanJing.env.sample.sampler import Sampler
from XuanJing.enhancement.advantage import enhance_advantage
from XuanJing.learner.base import BaseLearner


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

        sampler = Sampler(actor=actor, env=env, args=args)

        run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results") / args.alg
        os.makedirs(str(run_dir), exist_ok=True)

        writer = tb.SummaryWriter(logdir=run_dir)
        tqdm_range = tqdm(range(args.num_episodes))
        for i_episode in tqdm_range:
            sample_data = sampler.sample_episode(n_episode=2)
            enhance_sample_data, episodes_reward = enhance_advantage(sample_data)
            torch.set_grad_enabled(True)
            agent.updata_parameter(enhance_sample_data)
            torch.set_grad_enabled(False)

            PipeLearner.save_logging(writer, agent.logging, i_episode)
            tqdm_range.set_postfix({"Episode": f"{i_episode}",
                                    "AverageEpisodeReward": f"{sum(episodes_reward) / len(episodes_reward)}"})
