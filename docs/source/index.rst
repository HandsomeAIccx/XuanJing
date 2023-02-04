.. XuanJing documentation master file, created by
   sphinx-quickstart on Thu Dec  8 21:58:10 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


**XuanJing** is a benchmark library of decision algorithms for reinforcement learning, imitation learning, multi-agent learning and planning algorithms.


Welcome to XuanJing!
====================================

In both supervised learning and reinforcement learning, the algorithm consists of two main components.
the **data** and the **update formula**. XuanJing abstracts these two parts, so that it is possible to train reinforcement learning algorithms in the same way as supervised learning.

This documentation is organized as followed:

.. toctree::
   :maxdepth: 1
   :caption: Contents:

.. toctree::
   :maxdepth: 1
   :caption: Installation

.. toctree::
   :maxdepth: 1
   :caption: Framework

   framework/actor

.. toctree::
   :maxdepth: 1
   :caption: Usages


.. toctree::
   :maxdepth: 1
   :caption: Algorithms

   algorithms/ucb
   algorithms/ddpg
   algorithms/ppo
   algorithms/trpo
   algorithms/ddpm
   algorithms/psro
   algorithms/be


FAQ
========




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
