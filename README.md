
# XuanJing

 [![GitHub stars](https://img.shields.io/github/stars/tinyzqh/XuanJing)](https://github.com/tinyzqh/XuanJing/stargazers) 
 [![GitHub forks](https://img.shields.io/github/forks/tinyzqh/XuanJing)](https://github.com/tinyzqh/XuanJingnetwork) 
 [![GitHub issues](https://img.shields.io/github/issues/tinyzqh/XuanJing)](https://github.com/tinyzqh/XuanJing/issues) 

**XuanJing** is a benchmark library of decision algorithms for reinforcement learning, imitation learning,
multi-agent learning and planning algorithms.

In both supervised learning and reinforcement learning, the algorithm consists of two main components.
: the data and the update formula.
XuanJing abstracts these two parts, so that it is possible to train reinforcement 
learning algorithms in the same way as supervised learning.

## Status

WIP. Not released yet.

## Table of Contents

- [FileFramework](#fileframework)
- [Install](#install)
- [Usage](#usage)
  - [Support](#support)  
  - [Example Readmes](#example-readmes)
- [Contributors](#contributors)
- [License](#license)
- [Citation](#citation)


## FileFramework


**Env** is in responsible for parallelizing and wrapping the environment.
The task of interacting with the environment falls to the **actor**. 
The data produced during the interaction between the actor and the environment 
is stored in the **buffer**(if needed.).
When an actor interacts with an environment, **learner** is in charge of managing the 
data and algorithms. **enhancement** is used to enhance the data in the buffer.
Model parameters are updated by the learner using data and **algorithms**.
**utils** are a class of useful functions.



## Install

TODO

## Usage

TODO

### Support

Supported algorithms are as following:

#### model free reinforcement learning

- [Deep Q-Network (DQN)](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
- [Double DQN](https://arxiv.org/pdf/1509.06461.pdf)
- [Dueling DQN](https://arxiv.org/pdf/1511.06581.pdf)
- [Proximal Policy Optimization (PPO)](https://arxiv.org/pdf/1707.06347.pdf)
- [Soft Actor-Critic (SAC)](https://arxiv.org/pdf/1812.05905.pdf)

#### model based reinforcement learning


#### Imitation Learning


#### planning algorithms

- [Regret Minimization in Games with Incomplete
Information (CFR)](https://proceedings.neurips.cc/paper/2007/file/08d98638c6fcd194a4b1e6992063e944-Paper.pdf)


### Example Readmes

To see how the specification has been applied, see the [example-readmes](example-readmes/).



## Contributors

This project exists thanks to all the people who contribute. 

<a href="https://github.com/tinyzqh/XuanJing/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=tinyzqh/XuanJing" />
</a>

Made with [contributors-img](https://contrib.rocks/preview?repo=tinyzqh%2FXuanJing).

## License

[MIT](LICENSE) © tinyzqh

## Citation

If you find XuanJing useful, please cite it in your publications.

```
@software{XuanJing,
  author = {Zhiqiang He},
  title = {XuanJing},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/tinyzqh/XuanJing}},
}
```