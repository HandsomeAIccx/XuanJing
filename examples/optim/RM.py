"""
Robbins-Monro Algorithm
"""

import math
import numpy as np
import matplotlib.pyplot as plt


def fun(x):
    return math.pow(x, 3) - 5


def RobbinsMonro(k):
    w_k = 0
    estimated_root = []
    observation_noise = []
    x_axis = []
    for i in range(1, k):
        noise = np.random.randn()
        w_k_1 = w_k - 1/k * (fun(w_k) + noise)
        estimated_root.append(w_k)
        observation_noise.append(noise)
        x_axis.append(i)
        w_k = w_k_1
    plt.subplot(211)
    plt.scatter(x_axis, estimated_root, s=30, facecolors='none', edgecolors='b')
    plt.plot(x_axis, estimated_root)
    plt.ylabel("Estimated Root W_k")
    plt.ylim((0, 2))

    plt.subplot(212)
    plt.scatter(x_axis, observation_noise, s=30, facecolors='none', edgecolors='b')
    plt.plot(x_axis, observation_noise)
    plt.ylabel("Observation Noise")
    plt.ylim((-3, 3))
    plt.show()


if __name__ == "__main__":
    RobbinsMonro(50)

