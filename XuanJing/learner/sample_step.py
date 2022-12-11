import os
import torch
import torch.optim
import numpy as np
from tqdm import tqdm
from pathlib import Path
import tensorboardX as tb

from XuanJing.env.sample.sampler import Sampler
from XuanJing.enhancement.next_state import enhance_next_state
from XuanJing.learner.base import BaseLearner


class PipeLearner(BaseLearner):
    @staticmethod
    def run(
            args,
            env,
            actor,
            agent
    ):
        PipeLearner.set_global_seeds(args.seed)

        sampler = Sampler(actor=actor, env=env, args=args)

        run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results") / args.alg
        os.makedirs(str(run_dir), exist_ok=True)

        writer = tb.SummaryWriter(logdir=run_dir)
        tqdm_range = tqdm(range(args.num_episodes))
        return_list = []
        for i_episode in tqdm_range:
            episodes_reward = 0
            done = False
            while not done:
                sampler.sample_step(n_step=1)
                sample_data = sampler.get_sampler_data()
                done = sample_data.get_value("done")
                reward = sample_data.get_value("reward")
                episodes_reward += reward
                torch.set_grad_enabled(True)
                agent.updata_parameter(sample_data)
                torch.set_grad_enabled(False)
            return_list.append(episodes_reward)
            tqdm_range.set_postfix({'episode': '%d' % (args.num_episodes / 10 * i_episode + i_episode + 1),
                                    'return': '%.3f' % np.mean(return_list[-10:])})

