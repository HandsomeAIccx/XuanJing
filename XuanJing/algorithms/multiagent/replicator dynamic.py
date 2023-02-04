"""
Ref: https://github.com/shanest/replicator_dynamic_examples
"""

import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt


class ReplicatorDynamic(object):
    def __init__(self, u):
        self.U = u  # 效用矩阵

    def fitness(self, i, policy):
        """
        # fit(i) = sum_j ( p_j * u(i, j) )
        我方选取动作i, 对手的策略为policy时, 我方的收益
        """
        return sum([policy[j] * self.U[i][j] for j in range(len(policy))])

    def avg_fitness(self, policy):
        """
        # avgfit = sum_j pj * fit(j)
        遍历策略, 看对手也是此策略下的平均收益
        """
        return sum([policy[j] * self.fitness(j, policy) for j in range(len(policy))])

    def rep(self, policy, t):
        """
        # policy has p1, ... , p_n-1, representing proportions of each strategy
        # pn = 1 - sum_i < n pi
        # pi = pi * ( fit(i) - avgfit )
        """
        all_policys = np.concatenate((policy, [1-sum(policy)]))
        return [policy[i] * (self.fitness(i, all_policys) - self.avg_fitness(all_policys)) for i in range(len(policy))]

    def rep_mut(self, pop_vec, t):
        """
        # replicator-mutator dynamics
        # pi = (sum_j fit(j) * pj * Q_ji ) - pi * avgfit
        """
        allpops = np.concatenate((pop_vec, [1 - sum(pop_vec)]))
        return [sum([self.fitness(j, allpops) * allpops[j] * Q[j][i] for j in range(len(allpops))]) -
                pop_vec[i] * self.avg_fitness(allpops)
                for i in range(len(pop_vec))]


def plot1d(fun):
    # call to visualize the dynamics for the 2-player games
    for y0 in np.arange(0.05, 0.95, .1):
        y = odeint(fun, [y0], t)
        plt.plot(y)
    plt.show()


def plot2d(fun):
    # call to visualize the dynamics for the 3-player games
    for s0 in np.arange(0.05, 0.95, .1):
        # 1-s0 to ensure that population vector lies in simplex
        for r0 in np.arange(0.05, 1-s0, .1):
            y0 = [s0, r0]
            y = odeint(fun, y0, t)
            plt.plot(y[:, 1], y[:, 0])
    plt.show()


if __name__ == "__main__":
    t = np.arange(0, 100, .1)
    # 1. rock-paper-scissors, you should see 'cycles'
    u_rsp = [[1, 2, 0],
             [0, 1, 2],
             [2, 0, 1]]
    RD_RSP = ReplicatorDynamic(u=u_rsp)
    plot2d(RD_RSP.rep)

    # 2. rock-paper-scissors with small mod
    # larger e makes 'inward spiral' easier to see
    # with this modification, the point [1/3 , 1/3 , 1/3] is globally stable
    # so, you should see all points 'spiraling in' towards 1/3, 1/3
    e = 0.1
    u_rsp_mod = [[1-e, 2, 0],
                 [0, 1-e, 2],
                 [2, 0, 1-e]]
    RD_RSP_MOD = ReplicatorDynamic(u=u_rsp_mod)
    plot2d(RD_RSP_MOD.rep)

    # 3
    Q_3 = [[0.95, 0.025, 0.025],
         [0.025, 0.95, 0.025],
         [0.025, 0.025, 0.95]]
    RD_Q_3 = ReplicatorDynamic(u=Q_3)
    plot2d(RD_Q_3.rep)

    # 4. prisoner's dilemma, you should see rapid convergence to 0 (all defect)
    u_pd = [[3, 1],
            [4, 2]]
    RD_PD = ReplicatorDynamic(u=u_pd)
    plot1d(RD_PD.rep)

    # 5.
    Q = [[0.95, 0.05],
         [0.05, 0.95]]
    RD_PD = ReplicatorDynamic(u=Q)
    plot1d(RD_PD.rep)

    # 6. hawkdove game
    # you should see rapid convergence to 1/2 (mixed pop of hawks and doves)
    u_hawk_dove = [[0, 3],
                   [1, 2]]
    RD_HawkDove = ReplicatorDynamic(u=u_hawk_dove)
    plot1d(RD_RSP.rep)

    # 7. stag hunt
    # you should see a 'knife edge': above a line, convergence to 1; below, convergence to 0
    u_sh = [[3, 3],
            [0, 4]]
    RD_SH = ReplicatorDynamic(u=u_sh)
    plot1d(RD_SH.rep)

    # 8.
    # 2-population replicator dynamic for 2x2x2 signaling game with only separating strategies

    # pop_vec has s1, r1 reflecting proportion playing sender1 and receiver1 strategies
    def rep(pop_vec, t):
        s1, r1 = pop_vec
        return [s1 * (r1 - (s1 * r1 + (1 - s1) * (1 - r1))), r1 * (s1 - (r1 * s1 + (1 - r1) * (1 - s1)))]


    # conflict of interest instead of common interest
    def rep_conflict(pop_vec, t):
        s1, r1 = pop_vec
        return [s1 * ((1 - r1) - (s1 * (1 - r1) + (1 - s1) * r1)), r1 * (s1 - (r1 * s1 + (1 - r1) * (1 - s1)))]


    t = np.arange(0, 100, .001)

    # y0 = [.3, .4]
    # y = odeint(rep_conflict, y0, t)
    # plt.plot(y[:,1], y[:,0])

    for s0 in np.arange(0.05, 0.95, .1):
        for r0 in np.arange(0.05, 0.95, .1):
            y0 = [s0, r0]
            y = odeint(rep_conflict, y0, t)
            plt.plot(y[:, 1], y[:, 0])

    plt.show()
