# Atari Zoo

The aim of this project is to disseminate deep reinforcement learning agents trained by a variety of algorithms, and to enable easy analysis, comparision, and visualization of them. The hope is to reduce friction for further 
research into understanding reinforcement learning agents. 
This project makes use of the excellent [Lucid](https://github.com/tensorflow/lucid) neural network visualization library, and integrates with the [Dopamine](https://github.com/google/dopamine) [model release](https://github.com/google/dopamine/tree/master/docs#downloads).

A paper introducing this work was published at the Deep RL workshop at NeurIPS 2018: [An Atari Model Zoo for Analyzing, Visualizing, and Comparing Deep Reinforcement Learning Agents](https://drive.google.com/open?id=0B_utB5Y8Y6D5OHdCbjFuYmtrZnBLVGkwZEdocU5YRVVLOFFZ).

## About

This software package is accompanied by a binary release of (1) frozen models trained on Atari games by a variety of deep reinforcement learning methods, and (2) cached gameplay experience of those agents in their
training environments, which is hosted online.

## Installation and Setup

Dependencies:
* [tensorflow](https://github.com/tensorflow/tensorflow) (with version >0.8)
* [lucid](https://github.com/tensorflow/lucid)
* [matplotlib](https://matplotlib.org/) for some visualiztions
* [moviepy](https://zulko.github.io/moviepy/) (optional for making movies) 
* [gym](https://github.com/openai/gym) (installed with support for Atari; optional for generating new rollouts)
* [tensorflow-onnx](https://github.com/onnx/tensorflow-onnx) (optional for exporting to [ONNX](https://onnx.ai/) format)

To install, run ```setup.py install``` after installing dependencies.

## Examples

```python

import atari_zoo
from atari_zoo import MakeAtariModel
from pylab import *

algo = "a2c"
env = "ZaxxonNoFrameskip-v4"
run_id = 1
tag = "final"
m = MakeAtariModel(algo,env,run_id,tag)()

# get observations, frames, and ram state from a representative rollout
obs = m.get_observations()
frames = m.get_frames()
ram = m.get_ram()

# visualize first layer of convolutional weights
session = atari_zoo.utils.get_session()

m.load_graphdef()
m.import_graph()

conv_weights = m.get_weights(session,0)
atari_zoo.utils.visualize_conv_w(conv_weights)
show()

```

From the command line you can run: ```python -m atari_zoo.activation_movie --algo rainbow --environment PongNoFrameskip-v4 --run_id 1 --output ./pong_rainbow1_activation.mp4```

## Notebooks

Example jupyter notebooks live in the notebook directory that give further examples of how this library can be used.

A [starter colab notebook](https://colab.research.google.com/github/uber-research/atari-model-zoo/blob/master/colab/AtariZooColabDemo.ipynb) enables you to check out the library without downloading and installing it.

## Web tools

* A tool for viewing videos of trained agents is available [here](https://uber-research.github.io/atari-model-zoo/video.html); note that it is possible to link to specific videos,
e.g. [https://uber-research.github.io/atari-model-zoo/video.html?algo=apex&game=Seaquest&tag=final&run=2](https://uber-research.github.io/atari-model-zoo/video.html?algo=apex&game=Seaquest&tag=final&run=2).

* A tool for viewing videos of trained agents alongside their neural activations is available [here](https://uber-research.github.io/atari-model-zoo/video2.html).

## Source code for training algorithms that produced zoo models

We trained four algorithms ourselves:

* [A2C](https://arxiv.org/abs/1602.01783) - we used the [baselines package from OpenAI](https://github.com/openai/baselines)
* [GA](https://arxiv.org/abs/1712.06567) - we used the [fast GPU implementation version released by Uber](https://github.com/uber-research/deep-neuroevolution)
* [ES](https://arxiv.org/abs/1703.03864) - we used the [fast GPU version released by Uber](https://github.com/uber-research/deep-neuroevolution)
* [Ape-X](https://arxiv.org/abs/1803.00933) - we used the [replication released by Uber](https://github.com/uber-research/ape-x)

We took trained final models from two algorithms (DQN and Rainbow) from the [Dopamine model release](https://ai.googleblog.com/2018/08/introducing-new-framework-for-flexible.html):

* [DQN](https://arxiv.org/abs/1312.5602) - [implementation here](https://github.com/google/dopamine)
* [Rainbow](https://arxiv.org/abs/1710.02298) - [implementation here](https://github.com/google/dopamine)

## Citation

To cite this work in publications, please use the following BibTex entry:

```
@inproceedings{
title = {An Atari Model Zoo for Analyzing, Visualizing, and Comparing Deep Reinforcement Learning Agents},
author = {Felipe Such, Vashish Madhavan, Rosanne Liu, Rui Wang, Pablo Castro, Yulun Li, Ludwig Schubert, Marc Bellemare, Jeff Clune, Joel Lehman},
booktitle = {Proceedings of the Deep RL Workshop at NeurIPS 2018},
year = {2018},
}
```

## Contact Information

For questions, comments, and suggestions, email [joel.lehman@uber.com](mailto:mailto:joel.lehman@uber.com).
