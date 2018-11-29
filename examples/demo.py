# Copyright (c) 2018 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

import sys

sys.path.append("..")

import atari_zoo
from atari_zoo import MakeAtariModel
from pylab import *

algo = "a2c"
env = "ZaxxonNoFrameskip-v4"
run_id = 1
m = MakeAtariModel(algo,env,run_id)()

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
