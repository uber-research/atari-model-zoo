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

import argparse
import numpy as np
import moviepy.editor as mpy
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
import gym
from lucid.optvis.render import import_model
from atari_zoo import MakeAtariModel
from atari_zoo.utils import get_session
from atari_zoo.utils import conv_activations_to_canvas
from atari_zoo.utils import fc_activations_to_canvas
from atari_zoo.utils import get_activation_scaling
import tensorflow as tf

def gather_activations(m,obs,activations_tensor,session,X_t,batch_size=200,add_observations=True):
    #gather activations over entire trajectory
    obs_idx = 0
    length = obs.shape[0]

    collected_reps = []

    while obs_idx < length:
        rep = session.run(activations_tensor,{X_t:obs[obs_idx:obs_idx+batch_size]})
        collected_reps.append(rep)
        obs_idx += batch_size

    #collate representations
    compiled_rep = {}
    for layer in range(len(rep)):
        collected = np.vstack([r[layer] for r in collected_reps])
        #print(collected.shape)
        layer_name = m.layers[layer]['name']
        compiled_rep[layer_name] = collected

    return compiled_rep

def activations_to_frames(m,activations):
    obs_idx = 0
    frames = []
    length = activations.shape[0]

    if len(activations.shape)==4:
        scaling = get_activation_scaling(m,activations)

    for obs_idx in range(length):
        if len(activations.shape)==4:
            frame = conv_activations_to_canvas(m,activations,padding=1,idx=obs_idx,scaling=scaling)
        elif len(activations.shape)==2:
            frame = fc_activations_to_canvas(m,activations,padding=1,idx=obs_idx)  
        frames.append(frame)
    return frames

def make_clips_from_activations(m,_frames,obs,activations_tensor,session,X_t,fps=60):
    clip_dict = {}
    activations = gather_activations(m,obs,activations_tensor=activations_tensor,
                                     session=session,X_t=X_t,batch_size=1)
    
    for layer_idx in range(len(m.layers)):
        layer_name = m.layers[layer_idx]['name']
        print(layer_name)
        frames = activations_to_frames(m,activations[layer_name])
        clip = mpy.ImageSequenceClip([frame*255 for frame in frames], fps=60)
        clip_dict[layer_name] = clip
        
    #create observation movie
    n_obs = m.native_activation_representation(obs)
    frames = activations_to_frames(m,n_obs)
    clip = mpy.ImageSequenceClip([frame*255 for frame in frames], fps=fps)
    clip_dict['observations'] = clip
    
    #create raw rollout movie
    clip = mpy.ImageSequenceClip([frame for frame in _frames], fps=60)
    clip_dict['frames'] = clip
    
    return clip_dict

def side_by_side_clips(clip1,clip2):
    #calculate size of background canvas
    total_size_x = clip1.size[0] + clip2.size[0]
    total_size_y = max(clip1.size[1],clip2.size[1])

    #create background canvas
    bg_clip = mpy.ColorClip(size=(total_size_x,total_size_y), color=(255,255,255))

    duration = clip2.duration

    #align clips on canvas
    clip1=clip1.set_position(pos=(0,"center"))
    clip2=clip2.set_position(pos=((total_size_x-clip2.size[0],"center")))

    clip_list = [bg_clip,clip1,clip2]

    #composite together
    cc = mpy.CompositeVideoClip(clip_list,(total_size_x,total_size_y)).subclip(0,duration)
    return cc
    

def _MakeActivationVideoOneLayer(m,clip_dict,layer_no):
    labels = ["conv1","conv2","conv3","fc","output"]
    scales = [1.5,2.0,2.0,0.5,1.5]

    #get game frames
    clip1 = clip_dict['frames']

    #get activations from one layer
    layer_name = m.layers[layer_no]['name']
    clip2 = clip_dict[layer_name]
    clip2_scale = scales[layer_no]
    clip2 = clip2.resize(clip2_scale)
    return side_by_side_clips(clip1,clip2)


def _MakeActivationVideo(m,clip_dict):
    composite_size = (550,1000)

    clip_list = []
    clip_list.append(mpy.ColorClip(size=composite_size, color=(255,255,255)))

    labels = ["obs","conv1","conv2","conv3","fc","output"]
    scales = [1.0, 1.5,2.0,2.0,0.5,1.5]

    x_pos = 350
    y_pos = 25
    padding = 50
    label_fontsize = 20

    layers = m.layers.copy()
    layers.insert(0,{'name':'observations'})

    for layer_idx in range(len(labels)):
        layer_name = layers[layer_idx]['name']
    
        #get clip and resize it
        clip = clip_dict[layer_name]
        clip = clip.resize(scales[layer_idx])
    
        #calculate where to place it
        _x_pos = x_pos - 0.5 * clip.size[0]
        _y_pos = y_pos
        clip = clip.set_position((_x_pos,_y_pos))
    
        txtClip = mpy.TextClip(labels[layer_idx],color='black', fontsize=label_fontsize)
        txtPos = (x_pos - 0.5 * txtClip.size[0],y_pos - txtClip.size[1])
        clip_list.append(txtClip.set_position(txtPos))
    
        #offset coordinates
        y_pos += clip.size[1]
        y_pos += padding
        clip_list.append(clip)
    
    duration = clip.duration

    clip_list.append(clip_dict['frames'].set_position((50,580)))
    #clip_list.append(clip_dict['observations'].set_position((0,50)))

    cc = mpy.CompositeVideoClip(clip_list,composite_size).subclip(0,duration)
    #cc.ipython_display()
    return cc 

"""
Take a model and create a dictionary of MoviePy clips
for all the activations of the NN given a cached evaluation.
"""
def MakeClipDict(m):
    tf.reset_default_graph()

    m.load_graphdef()
    m.import_graph()
    obs = m.get_observations()
    frames = m.get_frames()
    
    #get a tf session
    session = get_session()

    #create a placeholder input to the network
    X_t = tf.placeholder(tf.float32, [None] + m.image_shape)

    #now get access to a dictionary that grabs output layers from the model
    T = import_model(m,X_t,X_t)
    activations = [T(layer['name']) for layer in m.layers]
    
    clip_dict = make_clips_from_activations(m,frames,obs,activations,session=session,X_t=X_t,fps=60)

    return clip_dict

"""
Take a model and a layer number (0=conv1,1=conv2,2=conv3) and
generate a side-by-side video of agent and activations on that
layer.
"""
def MakeActivationVideoOneLayer(m,layer_no,out_file=None):
    clip_dict = MakeClipDict(m)
    clip = _MakeActivationVideoOneLayer(m,clip_dict,layer_no)

    if out_file!=None:
        clip.write_videofile(out_file)

    return clip
    


"""
Take a model m and generate a side-by-side video of agent and activations 
"""
def MakeActivationVideo(m,video_fn=None):
    clip_dict = MakeClipDict(m)
    clip = _MakeActivationVideo(m,clip_dict)

    if video_fn!=None:
        clip.write_videofile(video_fn)

    return clip

def main():
    """
    Generates an activation movie for a rollout with a particular model
    """

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--algo', help='choose from [es, a2c, dqn]', type=str,default="ga")
    parser.add_argument('--environment', type=str,default="SeaquestNoFrameskip-v4")
    parser.add_argument('--run_id',type=int,default=1)
    parser.add_argument('--output', type=str, default="output.mp4")

    args = parser.parse_args()
    
    m = MakeAtariModel(args.algo,args.environment,args.run_id)()

    cc = MakeActivationVideo(m)
    cc.write_videofile(args.output)

if __name__=="__main__":
    main()
