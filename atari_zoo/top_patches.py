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




# Code to extract images patches that maximally activate particular neurons
# in an Atari convnet. Note that right now this code is specifically fit
# to the Atari convnet structure, and would need adaptation and generalization
# to fit to arbitrary structures. This would likely be non-trivial because it
# requires some reflection on the structure of the network (reasoning about
# pooling / convs) to calculate receptive fields at particular layers, etc.

import sys

import tensorflow as tf
import lucid
import numpy as np
import atari_zoo
from atari_zoo import MakeAtariModel
from atari_zoo.rollout import generate_rollout

from lucid.misc.io import show
import lucid.optvis.objectives as objectives
import lucid.optvis.param as param
import lucid.optvis.transform as transform
import lucid.optvis.render as render

import atari_zoo.utils
from atari_zoo.utils import conv_activations_to_canvas
from atari_zoo.utils import fc_activations_to_canvas

from lucid.optvis.render import import_model
from matplotlib.pyplot import *

#from IPython import embed


# receptive field at conv3 is 36
# receptive field at conv2 is 20
# receptive field at conv1 is 8 (8x8 conv...)

def pad_image(image, padSize, pad_values=0.):
    """ 
    Function that pads an image on all 4 sides, each side having the same padding. 
    simulating the receptive field that can be larger than original image 
    image: shape (batch, h, w, c) or (h, w, c)
    padSize: integer. Number of pixels to pad each side
    pad_values: what value to pad it with 
    
    """
    if len(image.shape) == 4: # (batch, h, w, c)
        pads = ((0,0), (padSize,padSize),(padSize,padSize), (0,0))
    elif len(image.shape) == 3: # (h, w, c)
        pads = ((padSize,padSize),(padSize,padSize), (0,0))
    else: 
        raise ValueError('Unsupported representation shape {}'.format(image.shape))
    ret = np.pad(image, pads, 'constant', constant_values=pad_values)
    return ret

def get_obs_patch(observation, ii, jj, receptive_stride=(36,8), pad_each_side=4+2*4+1*8,plot=False):
    """ Function that get a patch from an observation matrix, according to 
    a (ii, jj) location at a layer higher up 
    observation: (batch, h, w, c), normally (batch, 84, 84, 4)
    ii: integer index in the h dimension
    jj: integer index in the w dimension
    receptive_stride: a tuple of (receptive field size, stride size) indicating from this higher-up 
        layer where (ii, jj) is located, the size of receptive field and stride into the observation.
        For networks used in this application, the three conv layers have, respectively, 
        (8,4), (20,8), (36,8)
        onto the original observation.
    pad_each_side: how much the observation should be padded, due to the fact that receptive field at
        some point expand outside of the original image. Because there have been 3 layers of conv, having
        filter sizes of 8, 4, and 3, strides of 2, 2, and 1. Under "same" padding as they do, the eventual
        padding is 4 + 4*2 + 1*2*4 = 20
    """
    repp = pad_image(observation, pad_each_side)  # pad to (112,112,4)
    
    (rec_size, stride) = receptive_stride

    # the field to look at in observation
    top = int(ii*stride-rec_size/2)
    bot = int(ii*stride+rec_size/2)
    left = int(jj*stride-rec_size/2)
    right = int(jj*stride+rec_size/2)
    #print('Before pad: ', top, bot, left, right)
    print('bottom left location in original obs: ({},{})'.format(bot, left))
    [new_top, new_bot, new_left, new_right] = [k+pad_each_side for k in [top,bot,left,right]]
    #print('After pad: ', new_top, new_bot, new_left, new_right)
    #figure(figsize=(10,4))
    if plot:
        for cc in range(observation.shape[-1]):
            subplot(101+observation.shape[-1]*10+cc)
            #print('bottom left location in padded obs: ({},{})'.format(bot+pad_each_side, left+pad_each_side))
            matshow(repp[new_top:new_bot,new_left:new_right,cc], fignum=0)
    #print(repp[new_top:new_bot,new_left:new_right,cc].shape)
    return repp[new_top:new_bot,new_left:new_right,observation.shape[-1]-1], (top, left)

def build_model_get_act(algo, env, run_id=1, tag='final', local=True, which_layer=2):
    """ Function that builds/loads a model given algorithm algo and environment env, etc.,
    and obtain activations at a specific layer.
    which_layer: the index into layers. 0->Conv1, 1->Conv2, 2->Conv3, 3->FC
    """
    # Activation map shapes: 
    # 0 Online/Conv/Relu (21, 21, 32)
    # 1 Online/Conv_1/Relu (11, 11, 64)
    # 2 Online/Conv_2/Relu (11, 11, 64)
    # 3 Online/fully_connected/Relu (512)
    # 

    #TODO
    # load model
    m = MakeAtariModel(algo, env, run_id, tag=tag)()
    nA = atari_zoo.game_action_counts[env]
    acts_shapes = [(0,21,21,32), (0,11,11,64), (0,11,11,64), (0,512),(0,nA)] 
    # getting frames, observations
    obs = m.get_observations()
    frames = m.get_frames()

    # get the flow ready from observation the the layer activation you want
    m.load_graphdef()
    #get a tf session
    session = atari_zoo.utils.get_session()
    #create a placeholder input to the network
    X_t = tf.placeholder(tf.float32, [None] + m.image_shape)
    #now get access to a dictionary that grabs output layers from the model
    T = import_model(m,X_t,X_t)
    
    # the activation tensor we want
    acts_T = T(m.layers[which_layer]['name'])
    try:
        acts = session.run(acts_T, {X_t: obs})
    except:
        # some models does not allow batch size > 1 so do it one at a time
        acts = np.empty(acts_shapes[which_layer])
        for obs_1 in obs:
            obs_1 = np.expand_dims(obs_1, axis=0)
            #rep_1 = session.run(rep_layer_T, {X_t: obs_1})
            rep_1 = session.run(acts_T, {X_t: obs_1})
            acts = np.append(acts, rep_1, axis=0)
    if m.channel_order=='NCHW':
        acts = np.transpose(acts, axes=[0,2,3,1])

    print('Layer {} {} activations obtained. Shape {}'.format(which_layer, 
                                m.layers[which_layer]['name'], acts.shape))   
    return obs, acts, frames

def plot_topN_patches(activations, observations, which_filter=38, which_layer=2, which='top',n=3,plot=True):
    """ Plot the things
    activations: activations across all observations. e.g. (2501, 11, 11, 64)
    which_filter: the filter of interest, integer between e.g. [0, 64) for conv3
    Top 3 and Bottom 3 are determined by the activation vaules in activations
    Plots are first on activations and then on specific observation patches
    """
   
    #last two are fc layers
    receptive_stride = [(8,4), (20,8), (36,8),(84,0),(84,0)][which_layer]
    pad_each_side = [4, 4+4*2, 4+4*2+1*8,0,0][which_layer]

    # Find the maximum value in each channel of activation
    acts_filter = activations[..., which_filter]  # e.g. (2501, 11, 11)
    max_per_sample = []
    for act in acts_filter:     # each (11,11)
        max_per_sample.append(act.max())
    max_per_sample = np.array(max_per_sample)  
    top3 = max_per_sample.argsort()[::-1][:n]
    #print(max_per_sample)
    bot3 = max_per_sample.argsort()[:n]
    rand3 = np.random.choice(len(max_per_sample), n)

    if which.startswith('top'):
        picks = top3
    elif which.startswith('bot'):
        picks = bot3
    elif which.startswith('rand'):
        picks = rand3
    else:
        raise ValueError('which={"top", "bot", "rand"}')

    def plot_things(picks,plot=True):
        patches = []
        bottleft = []
        #figure(figsize=(10,4))
        for cc, sample_pick in enumerate(picks):
            
            if len(activations.shape)==2: #fc
                rep_pick = np.zeros((5,5))
                rep_pick[0,0] = activations[sample_pick,which_filter]+1e-6
            else:
                rep_pick = activations[sample_pick,:,:,which_filter]
                [ii, jj] = [int(x) for x in np.where(rep_pick == np.max(rep_pick))]
            if plot:
                figure(0,figsize=(10,4))
                subplot(1,n,1+cc)
                imshow(rep_pick)
                title('Maximum activation loc: ({},{})'.format(ii,jj))
            
                figure(cc+2,figsize=(12,4))

            if len(activations.shape)==2: #fc:
                _patches=observations[picks[cc]]
                if plot:
                    figure()
                    for k in range(4):
                        subplot(141+k)
                        matshow(_patches[...,k],fignum=0)
                _bl = (0,0)
            else:
                _patches, _bl = get_obs_patch(observations[picks[cc]], ii, jj, receptive_stride, pad_each_side)
            patches.append(_patches)
            bottleft.append(_bl)
        return np.array(patches), bottleft

    if plot:
        gray()
    patches, bottleft = plot_things(picks,plot=plot)
    
    return patches, picks, bottleft

if __name__=='__main__':
    algos = ['a2c','es','ga','apex','rainbow','dqn']
    game_list_local = ['AmidarNoFrameskip-v4',
                       'AtlantisNoFrameskip-v4',
                       'KangarooNoFrameskip-v4',
                       'ZaxxonNoFrameskip-v4',
                       'AssaultNoFrameskip-v4',
                       'EnduroNoFrameskip-v4',
                       'SeaquestNoFrameskip-v4',
                       'AsterixNoFrameskip-v4',
                       'FrostbiteNoFrameskip-v4',
                       'SkiingNoFrameskip-v4',
                       'AsteroidsNoFrameskip-v4',
                       'GravitarNoFrameskip-v4',
                       'VentureNoFrameskip-v4']
    
    algo = algos[-1] 
    env = 'SeaquestNoFrameskip-v4' # sequest
    
    observations, activations = build_model_get_act(algo, env, which_layer=2)
    plot_top3_bot3_patches(activations, observations, which_filter=38, which_layer=2)

