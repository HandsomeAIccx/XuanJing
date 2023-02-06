import argparse
import random
import numpy as np
import torch
import copy
import torch.nn as nn
import gym

from torch import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(5, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)


def state2feature(state, legal_action):
    state_with_action = []
    for action in legal_action:
        state_copy = copy.deepcopy(state)
        state_with_action.append(np.hstack([state_copy, np.array(action)]))
    return torch.tensor(np.vstack(state_with_action), dtype=torch.float)


def computer_return(episode_buffer):
    mc_samples = []
    G = 0
    for s_a in reversed(episode_buffer):
        G += s_a[1]
        mc_samples.append([s_a[0], G / 200])
    return mc_samples


def train_episode(env, model, args):
    episode_buffer = []
    state = env.reset()
    while True:
        legal_action = [0, 1]
        state_feature = state2feature(state, legal_action)
        if np.random.random() > args.epsilon_greedy:
            action_idx = random.choice(list(range(len(legal_action))))
            action = legal_action[action_idx]
        else:
            with torch.no_grad():
                action_output = model(state_feature)
            action_idx = torch.argmax(action_output, dim=0)[0]
            action = legal_action[action_idx.item()]
        next_state, reward, done, info = env.step(action)
        episode_buffer.append([state_feature[action_idx], reward])
        state = next_state
        if done:
            episode_experience = computer_return(episode_buffer)
            print("episode reward", sum([i[1] for i in episode_buffer]))
            return episode_experience


def actor_sample(i, model, q, args, lock):
    try:
        train_env = gym.make("CartPole-v0")
        while True:
            episode_experience = train_episode(train_env, model, args)
            for experience in episode_experience:
                with lock:
                    q.put(experience)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Exception in worker process % {i}")
        raise e


def args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_actor", type=int, default=1)
    parser.add_argument("--epsilon_greedy", type=float, default=0.99)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--tensorboard_log", type=str, default="logs")
    args = parser.parse_known_args()[0]
    return args


class MCAgent(object):
    def __init__(
            self,
            model,
            optim
    ):
        self.model = model
        self.optimizer = optim

    def update_paramter(self, datas):
        x = torch.cat([data[0].unsqueeze(0) for data in datas], dim=0)
        y = torch.tensor([data[1] for data in datas]).unsqueeze(1)
        predict = self.model(x)

        loss = ((predict - y) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.model.parameters(), 40)
        self.optimizer.step()
        return loss.item()


def train(args):
    model = Model()
    model.share_memory()

    lock = mp.Lock()

    ctx = mp.get_context("spawn")
    buffer_q = ctx.Queue()
    actor_process = []
    for i in range(args.num_actor):
        actor = ctx.Process(
            target=actor_sample,
            args=(i, model, buffer_q, args, lock)
        )
        actor.start()
        actor_process.append(actor)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr
    )

    learner = MCAgent(model, optimizer)
    writer = SummaryWriter(args.tensorboard_log)
    learn_step = 0
    while True:
        buffer_q_size = buffer_q.qsize()
        if buffer_q_size > args.batch_size:
            with lock:
                datas = [buffer_q.get() for _ in range(args.batch_size)]
                loss = learner.update_paramter(datas)
                writer.add_scalar("loss", loss, learn_step)
            learn_step += 1
            print()


if __name__ == "__main__":
    train(args())