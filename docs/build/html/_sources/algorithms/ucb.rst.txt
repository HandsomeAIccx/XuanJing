.. _base:



UCB
======


不等式基础知识
----------------


令 :math:`X_{1}, \cdots, X_{n}` 为 :math:`n` 个独立同分布的随机变量，假设其均值为 :math:`\mu`,
方差为 :math:`\sigma^{2}=Var[x]`。那么这 :math:`n` 个变量的估计期望为：

.. math::

    \hat{\mu} = \frac{1}{n} \sum_{j=1}^{n} X_{j}


那么这个估计期望是否是一个无偏估计呢？

当 :math:`E[\hat{\mu}] = \mu` 时，我们说这个期望是一个无偏估计。此时我们验证一下是否是无偏估计:


.. math::
    E[\hat{\mu}] &= E[\frac{1}{n} \sum_{i=1}^{n}X_{i}] \\
    &= \frac{1}{n} \sum_{i=1}^{n} E[X_{i}] \\
    &= \frac{1}{n} n \mu \\
    &= \mu

所以它是一个无偏估计。

真实的均值 :math:`\mu` 和估计均值 :math:`\hat{\mu}` 之间大概相差多远呢？


估计均值 :math:`\hat{\mu}` 也有一个方差。这里我们回想方差的定义为：:math:`Var[Z] \triangleq E([Z - E[Z]])^{2}`。
方差有两个基本的性质：

.. math::
    Var(X + Y) &= Var(X) + Var(Y) \\
    Var(kX) &= k^{2}Var(X)

注意这里 :math:`X` 和 :math:`Y` 必须相互独立。之后我们可以计算一下 :math:`\hat{\mu}` 的期望:

.. math::
    Var(\hat{\mu}) &= Var(\frac{1}{n}\sum_{i=1}^{n}X_{i}) \\
    &= \frac{1}{n^{2}} \sum_{i=1}^{n} Var(X_{i}) \\
    &= \frac{1}{n^{2}} n \sigma \\
    & = \frac{\sigma}{n}



那用方差来衡量 :math:`\mu` 和 :math:`\hat{\mu}` 之间的差距的这种方式是否理想呢？但是在现实情况中，
人们往往关注的是 :math:`\mu` 和 :math:`\hat{\mu}` 严重偏离情况发生的概率。从方差这个衡量指标，
我们看不出来是多数的情况下，误差极低，少数的情况错得很离谱。还是多数的情况 :math:`\hat{\mu}` 是可以接受的，
从来也不犯严重的错误。后者是我们理想的估计量。

因此，不采用方差来度量 :math:`\mu` 和 :math:`\hat{\mu}` 之间的差距。我们定义一个阈值，当 :math:`|\hat{\mu} - \mu| \geq \epsilon` 时，
我们认为这个阈值是不可接受的。因此，实际上我们期望计算的概率如下：

.. math::
    P(\hat{\mu} \geq \mu + \epsilon) \\
    P(\hat{\mu} \leq \mu - \epsilon) \\
    P(|\hat{\mu} - \mu| \leq \epsilon )


这三个概率被称作为尾概率。:math:`P(\hat{\mu} \geq \mu + \epsilon)` 是右尾概率(upper tail probalility),
:math:`P(\hat{\mu} \leq \mu - \epsilon)`是左尾概率(lower tail probability)，
:math:`P(|\hat{\mu} - \mu| \leq \epsilon )` 是双尾概率(two-sided tail probability)。

我们通常无法直接计算尾概率的值，但是我们可以计算尾概率的界。

马尔可夫不等式
***************

定理(马尔可夫不等式)：设 :math:`(\Omega, A, P)` 为概率空间，:math:`X` 为非负实值的随机变量，且 :math:`X` 定义在 :math:`\Omega` 上，
:math:`\epsilon > 0` 为一个常数，则有：

.. math::
    P(X \geq \epsilon) \leq \frac{E[x]}{\epsilon}

证明过程如下：

随机变量 :math:`X` 若是个连续分布，则其期望可以表示为 :math:`E(X) = \int_{0}^{\infty}X P(X) dx`。因此有：

.. math::
    E(X) &= \int_{0}^{\infty}X P(X) dx \\
    & \geq \int_{\epsilon}^{\infty}X P(X) dx \\
    & \geq \epsilon \int_{\epsilon}^{\infty} P(X) dx = \epsilon P(X \geq \epsilon)


所以得证。对于离散变量同理也可以证明出来。

切比雪夫不等式
***************

马尔可夫不等式的一个直接推论就是切比雪夫不等式。

对于任意的随机变量 :math:`X` 和常数 :math:`\epsilon \geq 0` 有：

.. math::
    P(|X - E[X]| \geq \epsilon) \leq \frac{Var[X]}{\epsilon^{2}}

与马尔可夫不等式的区别在于，这里是对于任意的随机变量。证明过程如下：

令 :math:`(X - E[X])^{2} = Y` , 将其带入马尔可夫不等式中有：

.. math::
    P((X - E[X])^{2} \geq \epsilon^{2}) &\leq \frac{E[(X - E[X])^{2}]}{\epsilon^{2}} \\
     \rightarrow P(|X - E[X]| \geq \epsilon) &\leq \frac{Var[X]}{\epsilon^{2}}

矩母函数与切诺夫不等式
**********************

切比雪夫不等式中对于尾概率的上界的定义是比较松的。切比雪夫不等式中是将变量 :math:`(X - E[X])^{2}` 带入马尔可夫不等式中，
这里是取的平方，但是我们其实是可以取任意次方的，
如 :math:`(X - E[X])^{k}`，但是取任意次方并不能保证变量非负，
因此可以对其加一个绝对值 :math:`|X - E[X]|^{k}`，此时再代入到马尔可夫不等式中，就有：

.. math::
    P(|X - E[X]| \geq \epsilon) \leq \frac{E[|X - E[x]|^{k}]}{\epsilon^{k}}


此时就可以通过调整 :math:`k` 来取不同的界。但是具体取多少呢？这个计算会比较不方便。
切诺夫界的是一个矩母函数，矩母函数会有一些比较好的性质。

矩母函数的定义为：假设 :math:`X` 为一个随机变量，若存在 :math:`h \geq 0`,
使得对于任意 :math:`\lambda \in [0, h]`, :math:`E[e^{\lambda x}]` 均存在，
则称$X$存在矩母函数(MGF), 记作 :math:`M_{x}(\lambda)`, 定义式为:

.. math::
    M_{x}(\lambda) \triangleq E[e^{\lambda x}]


它有一些比较好的性质，像它的:math:`i` 阶导数会等于 :math:`X` 的 :math:`i` 次方。

如果随机变量 :math:`X`满足 :math:`E[e^{\lambda x}] = \infty, \forall \lambda > 0`,
则称之为重尾，否则称之为轻尾。


在切比雪夫不等式中，是将 :math:`(X - E([X]))^{2}` 带入到马尔可夫不等式中，
但是在切诺夫不等式中，是将 :math:`e^{\lambda (x - \mu)}` 带入到马尔可夫不等式中去的。因此有：

.. math::
    P[(X - \mu) \geq \epsilon] &= P[e^{\lambda (x - \mu)} \geq e^{\lambda \epsilon}] \\
    &\leq \frac{E[e^{\lambda(x - \mu)}]}{e^{\lambda \epsilon}}


上述不等式对于任意的 :math:`\lambda \in [0, h]` 都成立。我们的目标是推一个尾概率的界，那么 :math:`\lambda` 取多少比较好呢？
由于是取一个界，那么肯定是希望这个界越紧越好，那么就是对 :math:`\frac{E[e^{\lambda(x - \mu)}]}{e^{\lambda \epsilon}}` 去取一个下确界。
这个界其实就是切诺夫界。

切诺夫界定义
******************

对于任意 :math:`X`，假设其均值存在，且为 :math:`\mu`, 并且其矩母函数 :math:`M_{x}(\lambda), \lambda \in [0, h]` 存在，则 :math:`X` 的切诺夫界定义为：

.. math::
    P[(x - \mu) \geq \epsilon] \leq inf_{\lambda \in [0, h]} \frac{E[e^{\lambda(x - \mu)}]}{e^{\lambda \epsilon}}


例：计算正态分布的切诺夫界：

.. math::
    M_{x}(\lambda) &= E[e^{\lambda x}] \\
    & = \int_{-\infty}^{+\infty} e^{\lambda x} \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{(x - \mu)^{2}}{2 \sigma^{2}}} dx \\
    & =  \int_{-\infty}^{+\infty}  \frac{1}{\sqrt{2\pi}\sigma} e^{(-\frac{[x - (\mu + \lambda \sigma^{2})]^{2}}{2 \sigma^{2}})} e^{\mu \lambda + \frac{\sigma^{2}\lambda^{2}}{2}} dx \\
    & = e^{(\mu \lambda + \frac{\sigma^{2} \lambda^{2}}{2})}


显然 :math:`M_{x}(\lambda)` 对于任意 :math:`\lambda \in R` 均有定义。所以:

.. math::
    & \inf_{\lambda \in R} \frac{E[e^{\lambda (x - \mu)}]}{e^{\lambda \epsilon}} \\
    & = \inf_{\lambda \in R} \frac{ e^{\frac{{\sigma^{2} \lambda^{2}} }{2}}}{e^{\lambda \epsilon}} \\
    & = \inf_{\lambda \in R} e^{\frac{\sigma^{2} \lambda ^{2}}{2} - \lambda \epsilon}


上式是一个凸函数，因此，直接对其求导数即可得到最小值的解为 :math:`\lambda^{*} = \frac{\epsilon}{\sigma^{2}}` 。
带入公式可以得出 :math:`\inf_{\lambda \in R} e^{\frac{\sigma^{2} \lambda ^{2}}{2} - \lambda \epsilon} = e^{-\frac{\epsilon^{2}}{2 \sigma}}`。

故利用切诺夫界得到的高斯分布的尾概率界为：

.. math::
    P[(x - \mu) \geq \epsilon] \leq e^{-\frac{\epsilon^{2}}{2 \sigma^{2}}}


次高斯性与霍夫丁界
*******************



上置信界
**********

上置信界(Upper Confidence Bound, UCB)算法是基于霍夫丁不等式推倒而来的一个算法。霍夫丁不等式说的是：
对于 :math:`n` 个独立同分布的随机变量 :math:`X_{1} , \cdots, X_{n}` , 取值范围为[0, 1]，其期望经验为 :math:`\hat{x}`。

.. math::
    \hat{x} = \frac{1}{n} \sum_{j=1}^{n} X_{j}

霍夫丁不等式描述的是：

.. math::
    P(E[X] \geq \hat{x_{t}} + u) \leq e^{-2nu^{2}}



对于多臂老虎机问题来说，每个臂都是一个独立同分布的随机变量。经验期望就是 :math:`\hat{Q}(a_{t})` ,
对应不等式中的 :math:`\hat{x}_{t}` 。不等式中的参数 :math:`u = \hat{U}(a_{t})` 代表不确定性度量。
给定一个概率 :math:`p = e^{-2 N(a_{t})U(a_{t})^{2}}`, 根据上述不等式，
可以得到 :math:`Q(a_{t}) \leq \hat{Q}(a_{t}) + \hat{U}(a_{t})` 至少以1-p的概率成立。那么当p很小时，
:math:`Q(a_{t}) \leq \hat{Q}(a_{t}) + \hat{U}(a_{t})` 就以很大的概率成立，
此时 :math:`\hat{Q}(a_{t}) + \hat{U}(a_{t})` 就是期望奖励上界。
此时上置信界算法便选取期望奖励上界最大的动作，
即 :math:`a_{t} = argmax_{a \in A} [\hat{Q}(a_{t}) + \hat{U}(a_{t})]` 。
因为 :math:`p = e^{-2 N(a_{t})U(a_{t})^{2}}`，所以 :math:`\hat{U}(a_{t}) = \sqrt{\frac{-log p}{ 2 N(a_{t})}}`。

也就是说，设定一个概率之后，每次决策先计算出期望奖励上界，并以很小的概率 :math:`p` 超过这个上界，接着选出期望奖励上界最大的拉杆。