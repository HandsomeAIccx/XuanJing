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
        for i_episode in tqdm(range(args.num_episodes)):
            sampler.sample_episode(n_episode=1)
            sample_data = sampler.get_sampler_data()
            enhance_sample_data = enhance_advantage(sample_data)
            torch.set_grad_enabled(True)
            agent.updata_parameter(enhance_sample_data)
            torch.set_grad_enabled(False)

            PipeLearner.save_logging(writer, agent.logging, i_episode)
            PipeLearner.save_logging(writer, sampler.logging, i_episode)

