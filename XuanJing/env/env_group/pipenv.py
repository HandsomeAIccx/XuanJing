import gym

from XuanJing.env.vector.vecbase import VectorEnv


def get_environment(args):
    """
    # TODO 环境的输入输出维度，sample函数等功能。基于env和给定网络类型直接实例化网络。
    # TODO env seed, env id的设定。
    # TODO env 的并行。
    """
    envs = VectorEnv([lambda: gym.make(args.task) for _ in range(args.env_num)])
    return envs


if __name__ == "__main__":
    pass