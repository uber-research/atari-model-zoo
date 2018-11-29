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
import lucid
import atari_zoo
from atari_zoo import MakeAtariModel

from lucid.misc.io import show
import lucid.optvis.objectives as objectives
import lucid.optvis.param as param
import lucid.optvis.transform as transform
import lucid.optvis.render as render

from lucid.optvis.param.color import to_valid_rgb
from lucid.optvis.param.spatial import naive, fft_image
import pylab
from lucid.optvis.objectives import wrap_objective, Objective
import matplotlib
import numpy as np
from lucid.misc.io import load, save, show
from lucid.misc.io.showing import images

#call to create raw image
def image(shape, add_noise=False):
  if add_noise:
    raw_frames = lucid.optvis.param.spatial.naive(shape, sd=0.5)
  else:
    raw_frames = lucid.optvis.param.spatial.naive(shape)
  processed_frames = tf.nn.sigmoid(raw_frames)
  return processed_frames

#if you want to only optimize current frame and leave past 3 frames to be zero
def only_current_frame(shape):
    shape_1 = shape[:-1]+[1,]
    
    shape_2 = shape[:]
    shape_2[-1] -= 1
    
    print(shape_1,shape_2)
    
    current_frame = lucid.optvis.param.spatial.naive(shape_1)
    zero_frames = tf.zeros(shape_2)
       
    processed_current = tf.nn.sigmoid(current_frame)
    processed_frames = tf.concat([zero_frames,processed_current],-1)
    return processed_frames

#create lucid objective functions that work with different channel orderings (
@wrap_objective
def channel(layer, n_channel, ordering="NHWC"):
  """Tensor-order aware version of channel lucid objective"""
  if ordering=='NCHW':
    return lambda T: tf.reduce_mean(tf.transpose(T(layer),perm=[0,2,3,1])[...,n_channel])
  else:
    return lambda T: tf.reduce_mean(T(layer)[..., n_channel])

#an L2 penalty only for a specific channel of the input image
@wrap_objective
def L2c(layer="input", constant=0, epsilon=1e-6, batch=None,channel=0):
  """L2 norm of layer. Generally used as penalty."""
  if batch is None:
    return lambda T: tf.sqrt(epsilon + tf.reduce_sum((T(layer)[...,channel] - constant) ** 2))
  else:
    return lambda T: tf.sqrt(epsilon + tf.reduce_sum((T(layer)[batch,...,channel] - constant) ** 2))

@wrap_objective
def direction_cossim(layer, vec, ordering="NHWC"):
  """Visualize a direction (cossine similarity)"""
  def inner(T):
    if ordering=='NCHW':
        _layer = T(layer)
    else:
        _layer = tf.transpose(T(layer),perm=[0,2,3,1])
        
    act_mags = tf.sqrt(tf.reduce_sum(_layer**2, -1, keepdims=True))
    vec_mag = tf.sqrt(tf.reduce_sum(vec**2))
    
    mags = act_mags * vec_mag
    return tf.reduce_mean(_layer * vec.reshape([1, 1, 1, -1]) / mags)
    
  return inner

@wrap_objective
def direction_neuroncossim(layer, vec, ordering="NHWC"):
  """Visualize a direction (cossine similarity)"""
  def inner(T):
    if ordering=='NCHW':
        _layer = T(layer)
    else:
        _layer = tf.transpose(T(layer),perm=[0,2,3,1])
        
    act_mags = tf.sqrt(tf.reduce_sum(_layer[:,5:6,5:6,:]**2, -1, keepdims=True))
    vec_mag = tf.sqrt(tf.reduce_sum(vec**2))
    
    mags = act_mags * vec_mag
    return tf.reduce_mean(_layer[:,5:6,5:6,:] * vec.reshape([1, 1, 1, -1]) / mags)
    
  return inner 


def make_regularization(L1=0.0,L2=0.0,TV=0.0):
    return -L1*objectives.L2()-L2*objectives.L2()-TV*objectives.total_variation()


def visualize_neuron(algo='apex',env='SeaquestNoFrameskip-v4',run_id=1,tag="final",param_f=lambda: image([1,84,84,4]),do_render=False,
                     transforms=[transform.jitter(3),],layer_no=0,neuron=0,regularization=0,**params):
    tf.reset_default_graph()
    
    m = MakeAtariModel(algo,env,run_id,tag,local=False)()
    m.load_graphdef()
   
    if(m.layers[layer_no]['type']=='dense'):
        obj = objectives.channel(m.layers[layer_no]['name'],neuron)
    else:
        obj = channel(m.layers[layer_no]['name'],neuron,ordering=m.channel_order)

    out = optimize_input(obj+regularization,m,param_f,transforms,do_render=do_render,**params)
    return out


#differentiable image parameterizations
from tensorflow.contrib import slim
import numpy as np

#CPPN setup
def composite_activation(x):
  x = tf.atan(x)
  # Coefficients computed by:
  #   def rms(x):
  #     return np.sqrt((x*x).mean())
  #   a = np.arctan(np.random.normal(0.0, 1.0, 10**6))
  #   print(rms(a), rms(a*a))
  return tf.concat([x/0.67, (x*x)/0.6], -1)


def composite_activation_unbiased(x):
  x = tf.atan(x)
  # Coefficients computed by:
  #   a = np.arctan(np.random.normal(0.0, 1.0, 10**6))
  #   aa = a*a
  #   print(a.std(), aa.mean(), aa.std())
  return tf.concat([x/0.67, (x*x-0.45)/0.396], -1)


def relu_normalized(x):
  x = tf.nn.relu(x)
  # Coefficients computed by:
  #   a = np.random.normal(0.0, 1.0, 10**6)
  #   a = np.maximum(a, 0.0)
  #   print(a.mean(), a.std())
  return (x-0.40)/0.58


def image_cppn(
    size,
    num_output_channels=1,
    num_hidden_channels=24,
    num_layers=8,
    activation_fn=composite_activation,
    normalize=False):
  r = 3.0**0.5  # std(coord_range) == 1.0
  coord_range = tf.linspace(-r, r, size)
  y, x = tf.meshgrid(coord_range, coord_range, indexing='ij')
  net = tf.expand_dims(tf.stack([x, y], -1), 0)  # add batch dimension

  with slim.arg_scope([slim.conv2d], kernel_size=1, activation_fn=None):
    for i in range(num_layers):
      in_n = int(net.shape[-1])
      net = slim.conv2d(
          net, num_hidden_channels,
          # this is untruncated version of tf.variance_scaling_initializer
          weights_initializer=tf.random_normal_initializer(0.0, np.sqrt(1.0/in_n)),
      )
      if normalize:
        net = slim.instance_norm(net)
      net = activation_fn(net)
      
    rgb = slim.conv2d(net, num_output_channels, activation_fn=tf.nn.sigmoid,
                      weights_initializer=tf.zeros_initializer())
  
  return rgb

def render_feature(
    cppn_f = lambda: image_cppn(84),
    optimizer = tf.train.AdamOptimizer(0.001),
    objective = objectives.channel('noname', 0),transforms=[]):
  vis = render.render_vis(m, objective, param_f=cppn_f, optimizer=optimizer, transforms=transforms, thresholds=[2**i for i in range(5,10)], verbose=False)
  #show(vis)
  return vis

#video rendering code...
from lucid.misc.io.serialize_array import _normalize_array
from lucid.misc.tfutil import create_session
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from IPython.display import clear_output, Image, display, HTML
import moviepy.editor as mpy
from lucid.modelzoo import vision_models
from lucid.misc.io import show, save, load
from lucid.optvis import objectives
from lucid.optvis import render

@wrap_objective
def all_activation(layer, batch=None):
  """Value of action minus average value of all actions"""
  if batch is None:
    return lambda T: tf.reduce_mean(T(layer))
  else:
    return lambda T: tf.reduce_mean(T(layer)[batch, ...])

cppn_default_f = lambda: image_cppn( 
          size=84, num_layers=8,num_hidden_channels=16,normalize=True, 
          activation_fn=relu_normalized, num_output_channels=4)

#composite_activation
#relu_normalized
#composite_activation_unbiased
def optimize_input(obj, model, param_f, transforms, lr=0.05, step_n=512,num_output_channels=4,do_render=False,out_name="out"):

  sess = create_session()

  # Set up optimization problem
  size = 84
  t_size = tf.placeholder_with_default(size, [])
  T = render.make_vis_T(
      model, obj, 
      param_f=param_f,
      transforms = transforms,
      optimizer=tf.train.AdamOptimizer(lr),
  )

  tf.global_variables_initializer().run()
 
  if do_render:
      video_fn = out_name + '.mp4'
      writer = FFMPEG_VideoWriter(video_fn, (size, size*4), 60.0)
  
  # Optimization loop
  try:
    for i in range(step_n):
      _, loss, img = sess.run([T("vis_op"), T("loss"), T("input")])

      if do_render:
          #if outputting only one channel...
          if num_output_channels==1:
              img=img[...,-1:] #print(img.shape)
              img=np.tile(img,3)
          else:
              #img=img[...,-3:]        
              img=img.transpose([0,3,1,2])
              img=img.reshape([84*4,84,1])
              img=np.tile(img,3)
          writer.write_frame(_normalize_array(img))
          if i > 0 and i % 50 == 0:
              clear_output()
              print("%d / %d  score: %f"%(i, step_n, loss))
              show(img)

  except KeyboardInterrupt:
    pass
  finally:
    if do_render:
        print("closing...")
        writer.close()
  
  # Save trained variables
  if do_render:
      train_vars = sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
      params = np.array(sess.run(train_vars), object)
      save(params, out_name + '.npy')
  
      # Save final image
      final_img = T("input").eval({t_size: 600})[...,-1:] #change size
      save(final_img, out_name+'.jpg', quality=90)

  out = T("input").eval({t_size: 84})
  sess.close()
  return out

 
