import random
import gym
import numpy as np
from collections import deque
# from human_preferences import HumanPreference
import os
import matplotlib.pyplot as plt
import datetime
import pathlib
import torch
from torch import nn

from torch import nn
import datetime
import torch
import cv2
import gym
import os
from time import sleep
import pathlib
import matplotlib.pyplot as plt
import argparse


class HumanPref(nn.Module):
    def __init__(self, obs_size, neuron_size=64):
        super(HumanPref, self).__init__()

        self.obs_size = obs_size
        self.neuron_size = neuron_size

        self.dense1 = nn.Linear(self.obs_size, self.neuron_size)
        self.dense2 = nn.Linear(self.neuron_size, 1)

        self.batch_norm = nn.BatchNorm1d(1)

    def forward(self, x1, x2=None):

        model1_couche1 = self.dense1(x1)
        model1_couche2 = torch.nn.functional.relu(model1_couche1)
        model1_couche3 = self.dense2(model1_couche2)
        model1_couche4 = self.batch_norm(model1_couche3)
        if x2 is None:
            return model1_couche4
        else:
            model2_couche1 = self.dense1(x2)
            model2_couche2 = torch.nn.functional.relu(model2_couche1)
            model2_couche3 = self.dense2(model2_couche2)
            model2_couche4 = self.batch_norm(model2_couche3)
            # output = nn.functional.softmax(torch.stack([model1_couche4, model2_couche4]), dim=0)
            p1_sum = torch.exp(torch.sum(model1_couche1)/len(x1))
            p2_sum = torch.exp(torch.sum(model2_couche4)/len(x2))
            p1 = p1_sum/torch.add(p1_sum, p2_sum)
            p2 = p2_sum / torch.add(p1_sum, p2_sum)
            return torch.stack([p1, p2])


class HumanPreference(object):
    def __init__(self, obs_size, action_size):
        self.trijectories = []
        self.preferences = []
        self.layer_count = 3
        self.neuron_size_init = 64
        self.batch_size_init = 10
        self.learning_rate = 0.00025
        self.obs_size = obs_size
        self.action_size = action_size
        self.neuron_size = obs_size ** 3

        self.loss_l = []

        self.create_model()

    def create_model(self):
        self.model = HumanPref(self.obs_size, self.neuron_size)
        self.criterion = nn.functional.binary_cross_entropy
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        self.model.train()
        if len(self.preferences) < 5:
            return

        batch_size = min(len(self.preferences), self.batch_size_init)
        r = np.asarray(range(len(self.preferences)))
        np.random.shuffle(r)

        min_loss = 1e+10
        max_loss = -1e+10
        lo = 0.0
        for i in r[:batch_size]:
            x0, x1, preference = self.preferences[i]

            pref_dist = np.zeros([2], dtype=np.float32)
            if preference < 2:
                pref_dist[preference] = 1.0
            else:
                pref_dist[:] = 0.5

            x0 = torch.from_numpy(np.asarray(x0)).float()
            x1 = torch.from_numpy(np.asarray(x1)).float()
            y = torch.from_numpy(pref_dist)
            y_hat = self.model(x0, x1)

            loss = self.criterion(y_hat, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if loss.item() > max_loss:
                max_loss = loss.item()
            elif loss.item() < min_loss:
                min_loss = loss.item()

            lo = loss.item()
        print("[ Loss: actual loss =", lo, " max =", max_loss, " min =", min_loss, "]")

        self.loss_l.append(lo)

    def predict(self, obs):
        self.model.eval()
        obs = torch.tensor([obs]).float()
        pred = self.model(obs)
        return pred.detach().numpy()

    def add_preference(self, o0, o1, preference):
        self.preferences.append([o0, o1, preference])

    def add_trijactory(self, trijectory_env_name,  trijectory):
        self.trijectories.append([trijectory_env_name, trijectory])

    def ask_human(self):

        if len(self.trijectories) < 2:
            return

        r = np.asarray(range(len(self.trijectories)))
        np.random.shuffle(r)
        t = [self.trijectories[r[0]], self.trijectories[r[1]]]

        envs = []
        for i in range(len(t)):
            env_name, trijectory = t[i]
            env = gym.make(env_name)
            env.reset()
            env.render()
            envs.append(env)

        cv2.imshow("", np.zeros([1, 1], dtype=np.uint8))

        print("Preference (1,2|3):")
        env_idxs = np.zeros([2], dtype=np.int32)
        preference = -1
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('1'):
                preference = 0
            elif key == ord('2'):
                preference = 1
            elif key == ord('3') or key == ord('0'):
                preference = 2

            if preference != -1:
                break

            for i in range(len(t)):
                envs[i].render()

                env_name, trijectory = t[i]
                obs, future_obs, action, done = trijectory[env_idxs[i]]
                envs[i].step(action)
                env_idxs[i] += 1
                if done or env_idxs[i] >= len(trijectory):
                    envs[i].reset()
                    env_idxs[i] = 0
            sleep(0.02)

        if preference != -1:
            os = []
            for i in range(len(t)):
                env_name, trijectory = t[i]
                o = []

                for j in range(len(trijectory)):
                    o.append(trijectory[j][1])

                os.append(o)

            self.add_preference(os[0], os[1], preference)

        cv2.destroyAllWindows()
        for i in range(len(envs)):
            envs[i].close()

        if preference == 0:
            print(1)
        elif preference == 1:
            print(2)
        elif preference != -1:
            print("neutral")
        else:
            print("no oppinion")


    def plot(self):
        x = np.arange(0, len(self.loss_l))
        y = np.asarray(self.loss_l)
        fig, ax = plt.subplots()
        ax.plot(y)
        ax.set_xlabel('epochs')
        ax.set_ylabel('loss')
        ax.set_title('Loss per epochs')

        datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = os.path.join(pathlib.Path().absolute(), 'plots', 'hp_model', 'hp_model' + datetime_str + ".png")
        plt.savefig(path)


class NN(nn.Module):
    def __init__(self, obs_size, action_size):
        super(NN, self).__init__()
        self.dense1 = nn.Linear(obs_size, 24)
        self.dense2 = nn.Linear(24, 24)
        self.dense3 = nn.Linear(24, 24)
        self.dense4 = nn.Linear(24, action_size)

    def forward(self, x):
        l1 = self.dense1(x)
        l2 = nn.functional.relu(l1)
        l3 = self.dense2(l2)
        l4 = nn.functional.relu(l3)
        l5 = self.dense3(l4)
        l6 = nn.functional.relu(l5)
        output = self.dense4(l6)
        return output


class DQNSolver:

    def __init__(self, observation_space, action_space, args):
        self.args = args
        self.exploration_rate = args.EXPLORATION_MAX

        self.scores = deque(maxlen=args.CONSECUTIVE_RUNS_TO_SOLVE)

        self.action_space = action_space
        self.memory = deque(maxlen=args.MEMORY_SIZE)

        self.rl_model = NN(observation_space, action_space)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.rl_model.parameters(), lr=args.LEARNING_RATE)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.rl_model(torch.tensor(state).float()).detach().numpy()
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < self.args.BATCH_SIZE:
            return
        batch = random.sample(self.memory, self.args.BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_pred = self.rl_model(torch.tensor(state_next).float()).detach().numpy()
                q_update = (reward + self.args.GAMMA * np.amax(q_pred))
            q_values = self.rl_model(torch.tensor(state).float()).detach().numpy()
            q_values[0][action] = q_update

            x = torch.from_numpy(state).float()
            y = torch.from_numpy(q_values).float()

            y_hat = self.rl_model(x)
            loss = self.criterion(y_hat, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()     # TODO
        self.exploration_rate *= self.args.EXPLORATION_DECAY
        self.exploration_rate = max(self.args.EXPLORATION_MIN, self.exploration_rate)

    def add_score(self, score):
        self.scores.append(score)

    def plot_score(self, mode, episodes):

        data = np.array(self.scores)
        x = []
        y = []
        x_label = "runs"
        y_label = "scores"
        for i in range(0, len(data)):
            x.append(int(i))
            y.append(int(data[i]))

        plt.subplots()
        plt.plot(x, y, label="score per run")

        average_range = len(x)
        plt.plot(x[-average_range:], [np.mean(y[-average_range:])] * len(y[-average_range:]), linestyle="--",
                 label="last " + str(average_range) + " runs average")

        if mode == "Human":
            plt.axvline(x=episodes/2, label="start of Human preference")

        if len(x) > 1:
            trend_x = x[1:]
            z = np.polyfit(np.array(trend_x), np.array(y[1:]), 1)
            p = np.poly1d(z)
            plt.plot(trend_x, p(trend_x), linestyle="-.", label="trend")

        plt.title(self.args.ENV_NAME)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        plt.legend(loc="upper left")

        datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = os.path.join(pathlib.Path().absolute(), 'plots', 'dqn', 'dqn_score_' + mode + '_' + datetime_str + ".png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()


def cartpole(args):
    env = gym.make(args.ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space, args)
    hp_model = HumanPreference(observation_space, action_space)
    run = 0
    episodes = 50
    for i in range(episodes):
        run += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0
        trijectory = []
        while True:
            step += 1
            # env.render()
            action = dqn_solver.act(state)
            state_next, reward, terminal, info = env.step(action)
            if args.mode == "self" or run < episodes / 2:
                reward = reward if not terminal else -reward
            else:
                reward = hp_model.predict(state_next)
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal)
            trijectory.append([state, state_next, action, terminal])
            state = state_next
            if terminal:
                print(
                    "Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
                hp_model.add_trijactory(args.ENV_NAME, trijectory)
                dqn_solver.add_score(step)
                break
            dqn_solver.experience_replay()

        if run % 5 == 0 and args.mode == "Human":
            hp_model.ask_human()
            hp_model.train()

    if args.mode == "Human":
        hp_model.plot()
    dqn_solver.plot_score(args.mode, episodes)


def rlhf_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="Human")
    parser.add_argument("--ENV_NAME", type=str, default="CartPole-v1")
    parser.add_argument("--GAMMA", type=float, default=0.95)
    parser.add_argument("--LEARNING_RATE", type=float, default=0.001)
    parser.add_argument("--MEMORY_SIZE", type=int, default=1000000)
    parser.add_argument("--BATCH_SIZE", type=int, default=20)
    parser.add_argument("--EXPLORATION_MAX", type=float, default=1.0)
    parser.add_argument("--EXPLORATION_MIN", type=float, default=0.01)
    parser.add_argument("--EXPLORATION_DECAY", type=float, default=0.995)
    parser.add_argument("--CONSECUTIVE_RUNS_TO_SOLVE", type=int, default=100)
    args = parser.parse_known_args()[0]
    return args


if __name__ == "__main__":
    rlhf = rlhf_args()
    cartpole(rlhf)
