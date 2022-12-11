.. _ppo:


PPO
==========

KL散度基础知识
-----------------

KL散度的数学形式为：

.. math::
    KL(P(X) || Q(X)) = \sum_{x \in X}[P(x) log \frac{P(x)}{Q(x)}] = E_{x \sim P(x)[log \frac{P(x)}{Q(x)}]}

KL散度具不对称性。

假设P为需要拟合的多峰分布，Q为用于拟合的变量，在拟合 :math:`KL(P, Q)` 与 :math:`KL(Q, P)` 时的不同？






