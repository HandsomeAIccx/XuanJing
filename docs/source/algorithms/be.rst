.. _be:


Bellman Equation
===================


贝尔曼公式描述了不同状态(state value)之间的关系.
给定一个随机的trajectory, 其Return可以表示为:

.. math::
    G_{t} &= R_{t+1} + \gamma R_{t+2} + \gamma^{2}R_{t+3} + \cdots \\
        & = R_{t+1} + \gamma(R_{t+2} + \gamma R_{t+3} + \cdots) \\
        & = R_{t+1} + \gamma G_{t+1}

对于状态值函数的定义可以表示为:

.. math::
    v_{\pi}(s) &= \mathbb{E}[G_{t}|S_{t} = s] \\
    & = \mathbb{E}[R_{t+1} + \gamma G_{t+1}|S_{t}=s] \\
    & = \mathbb{E}[R_{t+1}|S_{t}=s] + \gamma \mathbb{E}[G_{t+1}|S_{t}=s] \\

对于第一项 :math:`\mathbb{E}[R_{t+1}|S_{t}=s]`, 将其展开
我们有:

.. math::
    \mathbb{E}[R_{t+1}|S_{t}=s] &= \sum_{a}\pi (a|s) \mathbb{E}[R_{t+1}|S_{t}=s,A_{t}=a] \\
    & = \sum_{a} \pi(a|s) \sum_{r} p(r|s,a)r

这一项其实就是能够得到的及时奖励的mean.

对于第二项 :math:`\mathbb{E}[G_{t+1}|S_{t}=s]`:

.. math::
    \mathbb{E}[G_{t+1}|S_{t}=s] &= \sum_{s^{\prime}} \mathbb{E}[G_{t+1}|S_{t}=s,S_{t+1}=s^{\prime}]p(s^{\prime}|s) \\
    & = \sum_{s^{\prime}} \mathbb{E} [G_{t+1}|S_{t+1} = s^{\prime}] p(s^{\prime}|s) \\
    & = \sum_{s^{\prime}} v_{\pi}(s^{\prime}) p(s^{\prime}|s) \\
    & = \sum_{s^{\prime}} v_{\pi}(s^{\prime}) \sum_{a} p(s^{\prime} | s,a) \pi(a|s)

将上述两个公式带入, 我们就可以得到贝尔曼方程:

.. math::
    v_{\pi}(s) &= \mathbb{E}[R_{t+1}|S_{t}=s] + \gamma \mathbb{E}[G_{t+1}|S_{t}=s] \\
    & = \sum_{a} \pi(a|s) \sum_{r} p(r|s,a)r + \gamma \sum_{a} \pi(a|s) \sum_{s^{\prime}}p(s^{\prime}|s,a)v_{\pi}(s^{\prime}) \\
    & = \sum_{a} \pi(a|s) [\sum_{r}p(r|s,a)r + \gamma \sum_{s^{\prime}}p(s^{\prime}|s,a)v_{\pi}(s^{\prime})] \ \ \forall s \in S\\


上述公式对状态空间中的所有状态都成立. 因此我们会有一组这样的
式子, 我们联立这些式子, 就可以将其求解出来.
上述式子依赖很多概率, 像 :math:`\pi(a|s)` 其实就是
策略, 因此给定 :math:`\pi(a|s)` 求解方程的过程
就称为policy evaluation.
公式 :math:`p(r|s,a)` 和公式 :math:`p(s^{\prime}|s,a)` 表示dynamic model.


贝尔曼公式矩阵向量形式
>>>>>>>>>>>>>>>>>>>>>>


回顾一下贝尔曼公式:

.. math::
    v_{\pi}(s) &= \mathbb{E}[R_{t+1}|S_{t}=s] + \gamma \mathbb{E}[G_{t+1}|S_{t}=s] \\
    & = \sum_{a} \pi(a|s) \sum_{r} p(r|s,a)r + \gamma \sum_{a} \pi(a|s) \sum_{s^{\prime}}p(s^{\prime}|s,a)v_{\pi}(s^{\prime}) \\
    & = \sum_{a} \pi(a|s) [\sum_{r}p(r|s,a)r + \gamma \sum_{\prime}p(s^{\prime}|s,a)v_{\pi}(s^{\prime})] \ \ \forall s \in S\\

将其重写可以表示为:

.. math::
    v_{\pi}(s) = r_{\pi}(s) + \gamma \sum_{s^{\prime}}p_{\pi}(s^{\prime}|s)v_{\pi}(s^{\prime})

其中: :math:`r_{\pi}(s) \triangleq \sum_{a} \pi(a|s)\sum_{r}p(r|s,a)r`,
:math:`p_{\pi}(s^{\prime}|s) \triangleq \sum_{a}\pi(a|s)p(s^{\prime}|s,a)`

假设给状态标上下标之后我们可以得到 :math:`s_{i}(i=1,\cdots,n)`.
对于状态 :math:`s_{i}`, 贝尔曼方程有:

.. math::
    v_{\pi}(s_{i}) = r_{\pi}(s_{i}) + \gamma \sum_{s_{j}}p_{\pi}(s_{j}|s_{i})v_{\pi}(s_{j})

将所有的不等式写到一起, 我们可以得到贝尔曼方程的矩阵形式:

.. math::
    v_{\pi} = r_{\pi} + \gamma P_{\pi}v_{\pi}

其中:

.. math::
    & v_{\pi} = [v_{\pi}(s_{1}), \cdots, v_{\pi}(s_{n})]^{T} \in \mathbb{R}^{n} \\
    & r_{\pi} = [r_{\pi}(s_{1}), \cdots, r_{\pi}(s_{n})]^{T} \in \mathbb{R}^{n} \\
    & P_{\pi} \in \mathbb{R}^{n \times n}

对于 :math:`P_{\pi} \in \mathbb{R}^{n \times n}`, 其中 :math:`[P_{\pi}]_{ij}=p_{\pi}(s_{j}|s_{i})` 是状态矩阵.

求解贝尔曼方程状态值函数可以通过求解closed-form的方法:

已知:

.. math::
    v_{\pi} = r_{\pi} + \gamma P_{\pi}v_{\pi}

可得:

.. math::
    v_{\pi} = (I - \gamma P_{\pi})^{-1} r_{\pi}

但是求矩阵的逆计算量比较大. 所以在实际中
我们通常采用一种迭代的方法:

.. math::
    v_{k+1} = r_{\pi} + \gamma P_{\pi}v_{k}

当我们一直迭代下去, :math:`v_{k}` 就会收敛到真实的 :math:`v_{\pi}`.

Proof:

证明思路是定义 :math:`v_{k}` 到 :math:`v_{\pi}` 的误差
然后证明这个误差趋于0.

定义 :math:`\delta_{k} = v_{k} - v_{\pi}`, 那么 :math:`v_{k} = \delta_{k} + v_{\pi}`
将其带入

.. math::
    v_{k+1} = r_{\pi} + \gamma P_{\pi}v_{k}

可以得到:

.. math::
    \delta_{k+1} + v_{\pi} = r_{\pi} + \gamma P_{\pi}(\delta_{k} + v_{\pi})

将其整理一下有:

.. math::
    \delta_{k+1} &= - v_{\pi} + r_{\pi} + \gamma P_{\pi} \delta_{k} + \gamma P_{\pi} v_{\pi} \\
    &= \gamma P_{\pi} \delta_{k}

再将其展开有:

.. math::
    \delta_{k+1} &= \gamma P_{\pi}\delta_{k} = \gamma^{2}P_{\pi}^{2}\delta_{k-1} = \cdots \\
    &= \gamma^{k+1}P_{\pi}^{k+1}\delta_{0}

其中 :math:`0 \leq P_{\pi}^{k} \leq 1`. 另一方面 :math:`\gamma < 1`.
那么 :math:`\gamma^{k} \rightarrow 0`.
因此当 :math:`k \rightarrow \infty` 有  :math:`\delta_{k+1}=\gamma^{k+1}P_{\pi}^{k+1}\delta_{0} \rightarrow 0` .


贝尔曼最优公式
>>>>>>>>>>>>>>>>>>>>>>


贝尔曼最优公式是贝尔曼公式的一个特殊情况.
强化学习的目的就是寻找最优策略.

对于单步决策来说, 选择最优的策略可以直接
选取动作值比较大的那个:

.. math::
    a^{*} = argmax_{a}q_{\pi}(s_{1},a)

但是单步选取最优会是全局最优吗?其实迭代下去
是会的,会是最优的.

最优策略正式的定义为:

如果对于任意的一个状态 :math:`s`, 对于
任意的一个策略 :math:`\pi`, 当
:math:`v_{\pi^{*}}(s) \geq v_{\pi}(s)`
是, 我们称策略 :math:`\pi^{*}` 为最优策略.

那这样的一个最优策略是否存在呢?
这样的一个策略是否是唯一的?
策略是stochastic的还是deterministic的?
如何获得这样一个最优的策略?

想要回答上述问题, 需要从贝尔曼最优公式入手.
贝尔曼公式的定义为:

.. math::
    v_{\pi}(s) = \sum_{a} \pi(a|s) [\sum_{r}p(r|s,a)r + \gamma \sum_{s^{\prime}}p(s^{\prime}|s,a)v_{\pi}(s^{\prime})] \ \ \forall s \in S\\


在上述公式中, 策略 :math:`\pi` 是给定的.
如果是先求解策略 :math:`\pi` 期望获得最大化
值函数的过程就是贝尔曼最优公式:

.. math::
    v_{\pi}(s) &= max_{\pi} \sum_{a} \pi(a|s) [\sum_{r}p(r|s,a)r + \gamma \sum_{s^{\prime}}p(s^{\prime}|s,a)v_{\pi}(s^{\prime})] \ \ \forall s \in S\\
    & = max_{\pi} \sum_{a} \pi(a|s) q(s,a) \ \ s \in S.

将上述的elementwise form写成matrix-vetor form
可以表示为:

.. math::
    v = max_{\pi}(r_{\pi} + \gamma P_{\pi}v)

上述公式的右端项是一个最优化的问题. 但是我们需要
通过一个式子求解 :math:`v` 和 :math:`\pi` 两个
未知量的问题.

基于动作值函数得到的贝尔曼最优方程为:

.. math::
    v(s) = max_{\pi} \sum_{a} \pi(a|s)q(s,a)

很明显, 最优策略就是选择动作值函数最大的那个
动作, 其余动作选择概率都置零.

.. math::
    max_{\pi} \sum_{a} \pi(a|s)q(s,a) = max_{a \in \mathcal{A}(s)} q(s,a)

我们想要得到最优的 :math:`\pi`, 我们就需要固定
值函数 :math:`v` ,可以将其用函数的形式表示为:

.. math::
    f(v) := max_{\pi}(r_{\pi} + \gamma P_{\pi}v)

因此, 最优策略会是 :math:`v`的这样一个函数.
那贝尔曼最优公式:

.. math::
    v = f(v)

其中 :math:`f(v)` 中的 :math:`s` 对应的元素
是: :math:`[f(v)]_{s} = max_{\pi}\sum_{a}\pi(a|s)q(s,a), \ \ s \in S`

之后就是求解. 在求解之前我们需要先介绍一下压缩映射(Contraction mapping theorem)

Contraction mapping theorem
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

一些概念:

- 不动点(fixed point):

如果在一个集合 :math:`X`上有一个点 :math:`x` , :math:`x \in X` , 有一个映射, 或者
称之为函数  :math:`f: X \rightarrow X` ,
满足 :math:`f(x) = x` , 此时 :math:`x`
就被称作为不动点.

- 压缩映射(Contraction mapping):

如果一个函数满足以下不等式, 就被称为压缩映射:

.. math::
    ||f(x_{1}) - f(x_{2})|| \leq \gamma ||x_{1} - x_{2}||

其中 :math:`\gamma \in (0, 1)`.

举例像 :math:`x=f(x)=0.5x, \ \ x \in \mathbb{R}` 中
:math:`x = 0` 就是一个不动点, 函数 :math:`f(x)=0.5x` 就是
一个压缩映射(contraction mapping).

有了这两个概念之后就可以引出Contraction mapping theorem:

对于任意具有 :math:`x=f(x)` 形式的等式来说
如果 :math:`f` 是一个contraction mapping,
可以得到以下几点:

1. 一定存在一个fixed point :math:`x^{*}`, 满足
:math:`f(x^{*})=x^{*}`.
2. 这个不动点是唯一存在的.
3. 我们还知道如何求解这样一个算法: 是一个迭代式的
算法 :math:`x_{k+1}=f(x_{k})`. 当 :math:`k` 趋于无穷大时 :math:`x_{k} \rightarrow x^{*}` .
这个收敛速度是非常快的, 是指数收敛的.

Proof:

压缩映射定理依赖于柯西序列(Cauchy sequences),
对于一个极小值 :math:`\epsilon > 0` ,存在一个正整数
:math:`N` , 对于所有的 :math:`m,n > N`
我们有 :math:`||x_{m} - x_{n}|| \leq \epsilon` .

下面证明序列 :math:`\{x_{k}=f(x_{k-1})}_{k=1}^{\infty}` 是柯西序列
因此会收敛:

因为 :math:`f` 是一个压缩映射, 因此有:

.. math::
    ||x_{k+1}-x_{k}|| = ||f(x_{k})-f(x_{k-1})|| \leq \gamma ||x_{k}-x_{k-1}||

类似, 也能得到 :math:`||x_{k}-x_{k-1}|| \leq \gamma ||x_{k-1}-x_{k-2}||, \cdots`
因此有:

.. math::
    ||x_{k+1}-x_{k}|| & \leq \gamma ||x_{k}-x_{k-1}|| \\
    & \leq \gamma^{2} ||x_{k-1}-x_{k-2}|| \\
    & \cdots \\
    & \leq \gamma^{k} ||x_{1}-x_{0}||

因为 :math:`\gamma < 1` ,因此给定 :math:`x_{1},x_{0}`的情况下 :math:`||x_{k+1}-x_{k}||` 是指数收敛.
但是  :math:`||x_{k+1}-x_{k}||` 收敛, 并不代表 :math:`\{x_{k}\}` 能收敛, 因此需要去考虑
:math:`||x_{m}-x_{n}||` 对于任意的 :math:`m > n` 是否都是成立的:

.. math::
    ||x_{m}-x_{n}|| &= ||x_{m} - x_{m-1} + x_{m-1}- \cdots -x_{n+1}+x_{n+1}-x_{n}|| \\
    & \leq ||x_{m}-x_{m-1}|| + \cdots + ||x_{n+1}-x_{n}|| \\
    & \leq \gamma^{m-1}||x_{1}-x_{0}|| + \cdots + \gamma^{n} ||x_{1}-x_{0}|| \\
    & = \gamma^{n}(\gamma^{m-1-n}+\cdots+1)||x_{1}-x_{0}|| \\
    & \leq \gamma^{n}(1 + \cdots, +\gamma^{m-1-n} + \gamma^{m-n} + \gamma^{m-n+1}+\cdots)||x_{1}-x_{0}|| \\
    & = \frac{\gamma^{n}}{1-\gamma}||x_{1}-x_{0}||

因此, 对于任意的 :math:`\epsilon` , 我们总是可以找到一个正整数 :math:`N`
对于所有的 :math:`m,n > N` 有 :math:`||x_{m}-x_{n}|| < \epsilon` .
因此这个序列会是柯西序列, 那么也就会收敛到一个点
:math:`x^{*}=lim_{k \rightarrow \infty}x_{k}` .

因为是指数收敛, 所有在limit迭代过程中
会收敛到不动点.

第三点, 我们需要去证明这个不动点是唯一的
假设存在另一个不动点 :math:`x^{\prime}` 满足
:math:`f(x^{\prime})=x^{\prime}` .
然后, 我们有:

.. math::
    ||x^{\prime}-x^{*}|| = ||f(x^{\prime})-f(x^{*})|| \leq \gamma ||x^{\prime}-x^{*}||

因为 :math:`\gamma < 1` , 那么上述不等式当且仅当
:math:`x^{\prime}=x^{*}` 时成立.

回顾一下贝尔曼最优方程:

.. math::
    v = f(v) = max_{\pi}(r_{\pi} + \gamma P_{\pi}v) \\

因此, 接下来就需要去证明 :math:`f(v)` 是压缩映射.

Proof:

给定两个向量 :math:`v_{1},v_{2} \in \mathbb{R}^{|S|}` ,
假设 :math:`\pi_{1}^{*}=argmax_{\pi}(r_{\pi} + \gamma P_{\pi}v_{1})`
并且 :math:`\pi_{2}^{*}=argmax_{\pi}(r_{\pi} + \gamma P_{\pi}v_{2})`
由此有:

.. math::
    f(v_{1}) = max_{\pi}(r_{\pi} + \gamma P_{\pi}v_{1}) = r_{\pi_{1}^{*}} + \gamma P_{\pi_{1}^{*}}v_{1} \geq r_{\pi_{2}^{*}} + \gamma P_{\pi_{2}^{*}}v_{1} \\
    f(v_{2}) = max_{\pi}(r_{\pi} + \gamma P_{\pi}v_{2}) = r_{\pi_{2}^{*}} + \gamma P_{\pi_{2}^{*}}v_{2} \geq r_{\pi_{1}^{*}} + \gamma P_{\pi_{1}^{*}}v_{2}

将两式相减有:

.. math::
    f(v_{1}) - f(v_{2}) &= r_{\pi_{1}^{*}} + \gamma P_{\pi_{1}^{*}}v_{1} - (r_{\pi_{2}^{*}} + \gamma P_{\pi_{2}^{*}}v_{2}) \\
    & \leq r_{\pi_{1}^{*}} + \gamma P_{\pi_{1}^{*}}v_{1} - (r_{\pi_{1}^{*}} + \gamma P_{\pi_{1}^{*}}v_{2}) \\
    & = \gamma P_{\pi_{1}^{*}}(v_{1}-v_{2})

同理可以得到 :math:`f(v_{2})-f(v_{1}) \leq \gamma P_{\pi_{2}^{*}}(v_{2}-v_{1})` , 也就是:
:math:`f(v_{1})-f(v_{2}) \geq \gamma P_{\pi_{2}^{*}}(v_{1}-v_{2})` . 因此有:

.. math::
    \gamma P_{\pi_{2}^{*}}(v_{1}-v_{2}) \leq f(v_{1}) - f(v_{2}) \leq \gamma P_{\pi_{1}^{*}}(v_{1} - v_{2})

定义:

.. math::
    z = max\{|\gamma P_{\pi_{2}^{*}}(v_{1}-v_{2})|, |\gamma P_{\pi_{1}^{*}}(v_{1}-v_{2})|\} \in \mathbb{R}^{|S|}

可以得出:

.. math::
    |f(v_{1})-f(v_{2})| \leq z

假设 :math:`z_{i}` 是第 :math:`i` 个元素
:math:`p_{i}^{T}` 和 :math:`q_{i}^{T}`
是 :math:`P_{\pi_{1}^{*}}` 和 :math:`P_{\pi_{2}^{*}}`
的第 :math:`i` 行.

因此有:

.. math::
    |p_{i}^{T}(v_{1}-v_{2})| \leq p_{i}^{T}|v_{1}-v_{2}| \leq ||v_{1}-v_{2}||_{\infty}

对 :math:`q` 同理, 因此:

.. math::
    ||z||_{\infty} = max_{i}|z_{i}| \leq \gamma ||v_{1}-v_{2}||_{\infty}

联立可得:

.. math::
    ||f(v_{1})-f(v_{2})||_{\infty} \leq \gamma ||v_{1}-v_{2}||_{\infty}

因此能够得出函数 :math:`f(v)` 是具有收缩性质的.