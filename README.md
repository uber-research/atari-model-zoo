# Atari Zoo

The aim of this project is to disseminate deep reinforcement learning agents trained by a variety of algorithms, and to enable easy analysis, comparision, and visualization of them. The hope is to reduce friction for further 
research into understanding reinforcement leanring agents. 
This project makes use of the excellent [Lucid](https://github.com/tensorflow/lucid) neural network visualization library, and integrates with the [Dopamine](https://github.com/google/dopamine) [model release](https://github.com/google/dopamine/tree/master/docs#downloads).

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

Example jupyter notebooks also live in this directory that give further examples of how this library can be used. 
