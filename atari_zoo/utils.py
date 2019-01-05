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


import tensorflow as tf
import pylab
import numpy as np
import json
from lucid.misc.io.reading import read_handle

try:
    import tf2onnx
except:
    print('tf2onnx not installed, you will not be able to export to onnx')
    pass

"""
Helper function to load json from a url
(lucid.misc.io.reading.load chokes on a decoding issue)
"""
def load_json_from_url(url,cache=None,encoding='utf-8'):
    with read_handle(url,cache=cache) as handle:
        res = handle.read().decode(encoding=encoding)
        return json.loads(res)

"""
Helper function to generate a new session
"""
def get_session():
    tf.reset_default_graph()
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=0,
        intra_op_parallelism_threads=0)
    tf_config.gpu_options.allow_growth=True
    session = tf.Session(config=tf_config)
    return session

"""
Render a layer of conv weights to an
RGB numpy array canvas 
"""
def conv_weights_to_canvas(w):
    fx = w.shape[0]
    fy = w.shape[1]
    in_ch = w.shape[2]
    out_ch = w.shape[3]

    scale = 1
    padding = 1

    x_leap = (fx+padding)
    y_leap = (fy+padding)

    c_sz_x = padding + x_leap * in_ch
    c_sz_x *= scale
    c_sz_y = padding + y_leap * out_ch
    c_sz_y *= scale

    w_max = w.max()
    w_min = w.min()
    print(w_min,w_max)

    #first, cheap rescale
    w_scaled = (w-w_min)/(w_max-w_min)

    canvas = np.zeros((c_sz_x,c_sz_y,3))
    for i in range(in_ch):
        for j in range(out_ch):
            x_idx = padding + i*x_leap
            y_idx = padding + j*y_leap
        
            filt = w_scaled[:,:,i,j]
        
            canvas[x_idx:x_idx+fx,y_idx:y_idx+fy,0] = filt
            canvas[x_idx:x_idx+fx,y_idx:y_idx+fy,1] = filt
            canvas[x_idx:x_idx+fx,y_idx:y_idx+fy,2] = filt
        
    canvas = canvas.transpose([1,0,2])
    return canvas

"""
Use Matplotlib
"""
def visualize_conv_w(w,title=None,subsample=None):
    if subsample!=None:
        w=w[:,:,:,:subsample]
    canvas = conv_weights_to_canvas(w)
    #pylab.figure(figsize = (10,20))
    pylab.imshow(canvas)
    if title:
        pylab.title(title,fontsize=20)
    return canvas

#save model out to onnx format
def to_onnx(model,fname="./frozen_out.onnx",scope=""):
    tf.reset_default_graph()
    model.load_graphdef()
    model.import_graph(scope=scope)

    tf.import_graph_def(
            model.graph_def, {}, name=scope)

    graph = tf.get_default_graph()
    onnx_graph = tf2onnx.tfonnx.process_tf_graph(graph)

    inp_name = model.input_name+":0"
    out_name = model.layers[-1]['name']+":0"
    
    print(inp_name,out_name)
    model_proto = onnx_graph.make_model("", [inp_name], [out_name])

    with open(fname, "wb") as f:
        f.write(model_proto.SerializeToString())

    print("Done...")

#convert fc-level activations to a canvas representation
def fc_activations_to_canvas(m,act,scale=8,padding=1,width=32,idx=0):
    
    if len(act.shape)==2:
        act=act[idx]
    
    channels = act.shape[0]

    fx = fy = scale
    
    if width>channels:
        width=channels
    in_ch = width
    out_ch = int(channels / width)

    x_leap = (fx+padding)
    y_leap = (fy+padding)

    c_sz_x = padding + x_leap * in_ch
    #c_sz_x *= scale
    c_sz_y = padding + y_leap * out_ch
    #c_sz_y *= scale

    #print(c_sz_x,c_sz_y)

    a_max = act.max()
    a_min = act.min()
    #print(a_max,a_min)

    #first, cheap rescale
    a_scaled = (act-a_min)/(a_max-a_min)

    canvas = np.zeros((c_sz_x,c_sz_y,3))
    canvas[:,:,0]=1.0
    for i in range(in_ch):
        for j in range(out_ch):
            x_idx = padding + i*x_leap
            y_idx = padding + j*y_leap

            filt = a_scaled[i+j*width]

            canvas[x_idx:x_idx+fx,y_idx:y_idx+fy,0] = filt
            canvas[x_idx:x_idx+fx,y_idx:y_idx+fy,1] = filt
            canvas[x_idx:x_idx+fx,y_idx:y_idx+fy,2] = filt

    canvas = canvas.transpose([1,0,2])
    return canvas

def get_activation_scaling(model,act):

        act = model.canonical_activation_representation(act)
        #print("Processed shape",act.shape)

        act_max_ch = act.max((0,1,2))
        act_min_ch = act.min((0,1,2))
        return act_max_ch,act_min_ch

#convert conv-level activations to a canvas representation
def conv_activations_to_canvas(model,act,scale=1,padding=1,width=8,idx=0,scaling=None):

    act_max_ch = None
    act_min_ch = None

    if scaling!=None:
        act_max_ch,act_min_ch = scaling

    if len(act.shape)==4:
        #handle NCHW and NHWC
        act = model.canonical_activation_representation(act)
        #print("Processed shape",act.shape)

        act = act[idx]

    fx = act.shape[0]
    fy = act.shape[1]
    channels = act.shape[2]

    #no blank squares
    if width>channels:
        width = channels
        
        
    in_ch = width
    out_ch = int(channels / width)
    
    x_leap = (fx+padding)
    y_leap = (fy+padding)

    c_sz_x = padding + x_leap * in_ch
    c_sz_x *= scale
    c_sz_y = padding + y_leap * out_ch
    c_sz_y *= scale
    

    #global max/min
    a_max = act.max()
    a_min = act.min()

    #first, cheap rescale
    a_scaled = (act-a_min)/(a_max-a_min)

    canvas = np.zeros((c_sz_x,c_sz_y,3))
    canvas[:,:,0]=1.0
    for i in range(in_ch):
        for j in range(out_ch):
            x_idx = padding + i*x_leap
            y_idx = padding + j*y_leap

            if act_max_ch is None:
                filt = a_scaled[:,:,i+j*width]
            else:
                channel = i+j*width
                filt = (act[:,:,channel] - act_min_ch[channel])/(act_max_ch[channel]-act_min_ch[channel]+1e-8)

            #flip x & y
            filt = np.transpose(filt,[1,0])

            canvas[x_idx:x_idx+fx,y_idx:y_idx+fy,0] = filt
            canvas[x_idx:x_idx+fx,y_idx:y_idx+fy,1] = filt
            canvas[x_idx:x_idx+fx,y_idx:y_idx+fy,2] = filt


    canvas = canvas.transpose([1,0,2])
    return canvas

try:
	import moviepy.editor as mpy
	from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
except:
	print("Moviepy not installed, movie generation features unavailable.")

from lucid.misc.io.serialize_array import _normalize_array
import numpy as np

def MakeVideo(m,fps=60.0,skip=1,video_fn='./tmp.mp4'):
    obs = m.get_observations() 
    frames = m.get_frames() 
    size_x,size_y = frames.shape[1:3]   

    writer = FFMPEG_VideoWriter(video_fn, (size_y, size_x), fps)
    for x in range(0,frames.shape[0],skip):
        writer.write_frame(frames[x])
    writer.close()

def load_clip_from_cache(algo,env,run_id,tag="final",video_cache="."):

    i_video_fn ="%s/%s-%s-%d-%s.mp4" % (video_cache,algo,env,run_id,tag)

    return  mpy.VideoFileClip(i_video_fn)


def movie_grid(clip_dict,x_labels,y_labels,grid_sz_x,grid_sz_y,label_padding=50,padding=5,label_fontsize=20):
    key = list(clip_dict.keys())[0]
    exemplar = clip_dict[key]
    size_x,size_y = exemplar.size
    duration = exemplar.duration

    x_step = (size_x+padding)
    y_step = (size_y+padding)

    composite_size = (label_padding + x_step * grid_sz_x), (label_padding + y_step * grid_sz_y)

    #load in all the movie clips
    for _x in range(grid_sz_x):
        for _y in range(grid_sz_y):
            pos =(label_padding + _x*x_step,label_padding + _y*y_step)
            clip_dict[(_x,_y)] = clip_dict[(_x,_y)].set_position(pos)
            #clip.write_gif(o_video_fn)

    clip_list = []
    #add background clip
    clip_list.append(mpy.ColorClip(size=composite_size, color=(255,255,255)))

    #now add x and y labels
    l_idx = 0
    if y_labels != None:
        for label in y_labels:
            txtClip = mpy.TextClip(label,color='black', fontsize=label_fontsize).set_position((0,label_padding+y_step*l_idx+(y_step/2)))
            l_idx+=1
            clip_list.append(txtClip)

    l_idx = 0
    if x_labels != None:
        for label in x_labels:
            txtClip = mpy.TextClip(label,color='black', fontsize=label_fontsize).set_position((label_padding+x_step*l_idx,label_padding/2))
            l_idx+=1
            clip_list.append(txtClip)
    
    for key in clip_dict:
        clip_list.append(clip_dict[key])
    
    cc = mpy.CompositeVideoClip(clip_list,composite_size)
    return cc




def rollout_grid(env,algos,run_ids,tag='final',clip_resize=0.5,label_fontsize=20,out_fn="composite.mp4",video_cache=".",length=None):

    clip_dict = {}
    key = None
    for algo in algos:
        for run_id in run_ids:
            key = (algo,run_id)
            clip_dict[key] = load_clip_from_cache(algo,env,run_id,tag,video_cache).resize(clip_resize)
            
    exemplar = clip_dict[key]
    size_x,size_y = exemplar.size
    duration = exemplar.duration

    #labels for grid
    y_labels = [("R%d"% r) for r in run_ids] 
    x_labels= algos

    label_padding = 50
    padding = 5

    num_runs = len(run_ids)

    x_step = (size_x+padding)
    y_step = (size_y+padding)

    composite_size = (label_padding + x_step * len(algos), label_padding + y_step * num_runs)

    algo_idx = 0

    #load in all the movie clips
    for algo in algos:
        for run_id in run_ids:
            pos =(label_padding + algo_idx*x_step,label_padding + (run_id-1)*y_step)
            clip_dict[(algo,run_id)] = clip_dict[(algo,run_id)].set_position(pos)
            #clip.write_gif(o_video_fn)
        
            print(env,algo,run_id)
        
        algo_idx+=1
    

    clip_list = []
    #add background clip
    clip_list.append(mpy.ColorClip(size=composite_size, color=(255,255,255)))

    #now add x and y labels
    l_idx = 0
    for label in y_labels:
        txtClip = mpy.TextClip(label,color='black', fontsize=label_fontsize).set_position((0,label_padding+y_step*l_idx+(y_step/2)))
        l_idx+=1
        clip_list.append(txtClip)

    l_idx = 0
    for label in x_labels:
        txtClip = mpy.TextClip(label,color='black', fontsize=label_fontsize).set_position((label_padding+x_step*l_idx,label_padding/2))
        l_idx+=1
        clip_list.append(txtClip)

    
    for key in clip_dict:
        clip_list.append(clip_dict[key])
    
    cc = mpy.CompositeVideoClip(clip_list,composite_size)

    if length!=None:
        duration = length

    cc = cc.resize(1.0).subclip(0,duration)

    if out_fn != None:
        cc.write_videofile(out_fn)

    return cc,clip_dict
