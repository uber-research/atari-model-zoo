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


# Contains wrapper classes over Lucid that enable loading frozen graphs
# into the Lucid framework
import json
import lucid
from lucid.modelzoo.vision_base import Model
from lucid.misc.io.loading import load
from lucid.misc.io.reading import read, local_cache_path
import tensorflow as tf
import numpy as np
from pdb import set_trace as bb
import atari_zoo.config
from atari_zoo.config import datadir_local_dict,datadir_remote_dict,url_formatter_dict 
from atari_zoo import game_action_counts 
from atari_zoo.utils import *
import atari_zoo.log


"""
Basic RL model class that extends Lucid's Model class

Implements extra methods:

get_observations (Loads precomputed observations if they exist)
get_frames (Loads precomputed RGB frames if they exist)
get_ram (loads precomputed 128-integer RAM snapshots if they exist)
ram_state_to_bits (change 128-integer RAM into 1024-bit RAM)

Implements additional class variables:
channel_order: "NHWC" or "NCHW"
preprocess_style: 'tf' for tensorflow preprocessing, 'np' for numpy, 'dopamine' for dopamine-style
"""
class RL_model(Model):
    channel_order = "NHWC" #typical channel order
    dataset = 'RL'
    valid_run_range = (1,3)

    #if the model exposes other interesting layers
    #e.g. A2C exposes a value and a policy head
    additional_layers = {}

    #minutae -- whether atari preprocessing is done in
    #tensorflow or numpy; only difference is in implmentation
    #details of bilinear filtering downsampling; but of course
    #RL agents overfit to it!
    #
    #tf = tensorflow; np = numpy
    preprocess_style = 'tf'

    image_shape = [84, 84, 4]
    input_scale = 1.0
    image_value_range = (0, 1) 
    input_name = 'X_t'
    ph_type = 'float32'

    """ 
    overwrite input creating function to handle different
    datatypes; Dopamine models want uint8 placeholders
    """
    def create_input(self, t_input=None, forget_xy_shape=True):
        if t_input == None and self.ph_type=='uint8':
            t_input = tf.placeholder(tf.uint8,self.image_shape)

        #return super().create_input(t_input,forget_xy_shape)
        return super(RL_model,self).create_input(t_input,forget_xy_shape)

    #TODO integrate these file loads with Lucid's cache mechanism
    def get_log(self):
        fname = self.log_path+"_log.json"

        if(fname.find('http')!=-1):
            return load_json_from_url(fname)
        else:
            return json.load(open(fname))

    def get_checkpoint_info(self):
        return atari_zoo.log.load_checkpoint_info(self.log_path)

    def get_observations(self):
        fname = self.data_path
        return load(fname)['observations']

    def get_frames(self):
        fname = self.data_path
        return load(fname)['frames']

    def get_ram(self):
        fname = self.data_path
        return load(fname)['ram']

    def get_scores(self):
        fname = self.data_path
        return load(fname)['score']

    def get_representation(self):
        fname = self.data_path
        return load(fname)['representation']

    def get_episode_rewards(self):
        fname = self.data_path
        return load(fname)['ep_rewards']

    #TODO make more efficent
    def ram_state_to_bits(self,state):
        binary = ['{0:08b}'.format(k) for k in state]
        binary = ''.join(binary)
        return binary

    #what processing must be done to extract the
    #right distribution of actions from the output
    #of the network
    def get_action(self,model):
        raise NotImplementedError

    #transform weight tensor to be of canonical style
    def preprocess_weight(self,x):
        #default is identity
        return x
   
    #grab weights from model given current session 
    def get_weights(self,session,layer_no):
        weights_name = self.weights[layer_no]['name']
        weights = session.graph.get_tensor_by_name("import/%s:0" % weights_name)
        weights = self.preprocess_weight(weights)
        return session.run(weights)

    #transform activations into canonical tensor
    def canonical_activation_representation(self,act):
        if self.channel_order=='NHWC':
            return act
        else:
            #print("Current:",act.shape)
            return np.transpose(act,axes=[0,2,3,1])

    #transform activations into canonical tensor
    def native_activation_representation(self,act):
        if self.channel_order=='NHWC':
            return act
        else:
            return np.transpose(act,axes=[0,3,1,2])
  
#OpenAI's evolution strategy algorithm
class RL_ES(RL_model):
  weights = [
      {'name':'es/layer1/conv1/w'},
      {'name':'es/layer2/conv2/w'},
      {'name':'es/layer3/conv3/w'},
  ]

  layers = [
     {'type': 'conv', 'name': 'es/layer1/Relu', 'size': 32},
     {'type': 'conv', 'name': 'es/layer2/Relu', 'size': 64},
     {'type': 'conv', 'name': 'es/layer3/Relu', 'size': 64},
     {'type': 'dense', 'name': 'es/layer4/Relu', 'size': 512},
     {'type': 'dense', 'name': 'es/layer5/out/out', 'size':18}
   ]

  def preprocess_weight(self,x):
      return x[0]

  def get_action(self,model):
        policy = model(self.layers[-1]['name']) 
        action_sample = tf.argmax(policy, axis=-1)
        return action_sample

#Uber's Deep GA
class RL_GA(RL_model):
  layers = [
     {'type': 'conv', 'name': 'ga/conv1/relu', 'size': 32},
     {'type': 'conv', 'name': 'ga/conv2/relu', 'size': 64},
     {'type': 'conv', 'name': 'ga/conv3/relu', 'size': 64},
     {'type': 'dense', 'name': 'ga/fc/relu', 'size': 512},
     {'type': 'dense', 'name': 'ga/out/signal', 'size':18}
   ]

  weights = [
      {'name':'ga/conv1/w'},
      {'name':'ga/conv2/w'},
      {'name':'ga/conv3/w'},
  ]

  def get_action(self,model):
        policy = model(self.layers[-1]['name'])
        action_sample = tf.argmax(policy, axis=-1)
        return action_sample

  def preprocess_weight(self,x):
      return x[0]

#Ape-X (recent high-performing DQN variant)
class RL_Apex(RL_model):
  channel_order = "NCHW"

  #note: action_value/Relu also worth considering...
  layers = [
     {'type': 'conv', 'name': 'deepq/q_func/convnet/Conv/Relu', 'size': 32},
     {'type': 'conv', 'name': 'deepq/q_func/convnet/Conv_1/Relu', 'size': 64},
     {'type': 'conv', 'name': 'deepq/q_func/convnet/Conv_2/Relu', 'size': 64},
      {'type': 'dense', 'name': 'deepq/q_func/state_value/Relu', 'size': 512},
     {'type': 'dense', 'name': 'deepq/q_func/q_values', 'size':18}
   ]

  weights = [
      {'name':'deepq/q_func/convnet/Conv/weights'},
      {'name':'deepq/q_func/convnet/Conv_1/weights'},
      {'name':'deepq/q_func/convnet/Conv_2/weights'}
  ]
 
  def get_action(self,model):
        policy = model(self.layers[-1]['name']) #"a2c/policy/BiasAdd")
        action_sample = tf.argmax(policy, axis=-1)
        return action_sample

#DQN from dopamine model dump
class RL_DQN_dopamine(RL_model):
  #ph_type = 'uint8'
  input_scale = 255.0
  preprocess_style = 'dopamine'
  image_value_range = (0, 255) 
  input_name = 'Online/Cast'
  valid_run_range = (1,3)

  weights = [
      {'name':'Online/Conv/weights'},
      {'name':'Online/Conv_1/weights'},
      {'name':'Online/Conv_2/weights'}
  ]

  layers = [
     {'type': 'conv', 'name': 'Online/Conv/Relu', 'size': 32},
     {'type': 'conv', 'name': 'Online/Conv_1/Relu', 'size': 64},
     {'type': 'conv', 'name': 'Online/Conv_2/Relu', 'size': 64},
     {'type': 'dense', 'name': 'Online/fully_connected/Relu', 'size': 512},
     {'type': 'dense', 'name': 'Online/fully_connected_1/BiasAdd', 'size':18}
   ]
 
  def get_action(self,model):
        policy = model(self.layers[-1]['name']) 
        action_sample = tf.argmax(policy, axis=1)
        return action_sample

  def get_log(self):
    raise NotImplementedError
      #Integration with Dopamine log formatting not yet complete."

  def get_checkpoint_info(self):
    raise NotImplementedError
       #,"Dopamine models include only the final checkpoint."

#Rainbow (slightly older high-performing DQN variant)
class RL_Rainbow_dopamine(RL_model):
  #ph_type = 'uint8'
  valid_run_range = (1,5)
  preprocess_style = 'dopamine'
  input_scale = 255.0
  image_value_range = (0, 255) 
  #input_name = 'state_ph'
  input_name = 'Online/Cast'

  weights = [
      {'name':'Online/Conv/weights'},
      {'name':'Online/Conv_1/weights'},
      {'name':'Online/Conv_2/weights'}
  ]

  layers = [
     {'type': 'conv', 'name': 'Online/Conv/Relu', 'size': 32},
     {'type': 'conv', 'name': 'Online/Conv_1/Relu', 'size': 64},
     {'type': 'conv', 'name': 'Online/Conv_2/Relu', 'size': 64},
     {'type': 'dense', 'name': 'Online/fully_connected/Relu', 'size': 512},
     #{'type': 'dense', 'name': 'Online/fully_connected_1/BiasAdd', 'size':18}
     {'type': 'dense', 'name': 'Online/Sum', 'size':18}
   ]

  additional_layers={'c51':{'type':'dense','name:': 'Online/fully_connected_1/BiasAdd', 'size':18*51}}

 
  def get_action(self,model):
        policy = model(self.layers[-1]['name'])
        action_sample = tf.argmax(policy, axis=1)
        return action_sample

  def get_log(self):
    raise NotImplementedError
    #"Integration with Dopamine log formatting not yet complete."

  def get_checkpoint_info(self):
    raise NotImplementedError
    #"Dopamine models include only the final checkpoint."

#A2C -- policy gradient algorithm
class RL_A2C(RL_model):
  weights = [
      {'name':'a2c/conv1/weights'},
      {'name':'a2c/conv2/weights'},
      {'name':'a2c/conv3/weights'}
  ]

  layers = [
     {'type': 'conv', 'name': 'a2c/conv1/Relu', 'size': 32},
     {'type': 'conv', 'name': 'a2c/conv2/Relu', 'size': 64},
     {'type': 'conv', 'name': 'a2c/conv3/Relu', 'size': 64},
     {'type': 'dense', 'name': 'a2c/fc/Relu', 'size': 512},
     #TODO: enable accesing a2c's value head as well! 
     #{'type': 'dense', 'name': 'a2c/value/BiasAdd', 'size':18},
     {'type': 'dense', 'name': 'a2c/policy/BiasAdd', 'size':18}
   ]
  
  def get_action(self,model):
        policy = model(self.layers[-1]['name']) 
        rand_u = tf.random_uniform(tf.shape(policy))
        action_sample = tf.argmax(policy - tf.log(-tf.log(rand_u)), axis=-1)
        return action_sample

### Instantiate concrete models using python magic
class_map = {'ga':RL_GA,'es':RL_ES,'apex':RL_Apex,'a2c':RL_A2C,'dqn':RL_DQN_dopamine,'rainbow':RL_Rainbow_dopamine}

#helper utility to make new python model classes
def _MakeAtariModel(model_class,name,environment,model_path,run_id,algorithm,log_path,data_path):
    #find number of actions in this particular game
    num_actions = game_action_counts[environment]

    #change last layer size to reflect available actions
    #layers = model_class.layers.copy()
    layers = list(model_class.layers) #python2.7 compatibility
    layers[-1]['size']=num_actions

    #create inherited class with correct properties (hack?)
    return type('Atari'+name,(model_class,),{'model_path':model_path,'environment':environment,'layers':layers,'run_id':run_id,'algorithm':algorithm,'log_path':log_path,'data_path':data_path})

"""
Helper function to get paths to model, rollout data, and log
for a particular algo/env/run combo
"""
def GetFilePathsForModel(algo,environment,run_no,tag='final',local=False):

    #if loading off of local disk (rare; only for development)
    if local:
        data_root = datadir_local_dict[algo]
        if tag==None:
            model_path = "%s/%s/model%d.pb" % (data_root,environment,run_no)
            data_path = "%s/%s/model%d_rollout.npz" % (data_root,environment,run_no)
        else:
            model_path = "%s/%s/model%d_%s.pb" % (data_root,environment,run_no,tag)
            data_path = "%s/%s/model%d_%s_rollout.npz" % (data_root,environment,run_no,tag)

        log_path = "%s/checkpoints/%s_%d" % (data_root,environment,run_no)

    #otherwise if loading off the canonical remote server (most common)
    else:
        data_root = datadir_remote_dict[algo]
        if tag==None:
            model_path = "%s/%s/model%d.pb" % (data_root,environment,run_no)
            data_path = "%s/%s/model%d_rollout.npz" % (data_root,environment,run_no)
        else:
            model_path = "%s/%s/model%d_%s.pb" % (data_root,environment,run_no,tag)
            data_path = "%s/%s/model%d_%s_rollout.npz" % (data_root,environment,run_no,tag)

        if (algo,'remote') in url_formatter_dict:
            model_path = url_formatter_dict[(algo,'remote')](data_root,algo,environment,run_no)
   
        log_path = "%s/checkpoints/%s_%d" % (data_root,environment,run_no)

    return model_path,data_path,log_path

"""
Function to query for available checkpoints for a model
"""
def GetAvailableTaggedCheckpoints(algo,environment,run_no,local=False):
    _,_,log_path = GetFilePathsForModel(algo,environment,run_no,local=local)
    json_data = atari_zoo.log.load_checkpoint_info(log_path) 
    chkpoint_info = atari_zoo.log.parse_checkpoint_info(json_data)
    return chkpoint_info



"""
Function to load model from the model zoo

algo: Algorithm (ga,es,apex,a2c,dqn,rainbow)
environment: Atari gym environment (e.g. SeaquestNoFrameskip-v4)
run_no: which run of the algorithm
tag: which tag to search for (e.g. 1HR, human, 1B, final)
local: boolean, whether to get the model from a local archive or from the remote server
"""
def MakeAtariModel(algo,environment,run_no,tag='final',local=False):

    model_path,data_path,log_path = GetFilePathsForModel(algo,environment,run_no,tag,local)

    if atari_zoo.config.debug:
        print('Model path:',model_path)
        print('Data path:',data_path)
        print('Log path:',log_path)

    name = "%s_%s_%d_%s" % (algo,environment,run_no,tag)

    model_class = class_map[algo]
   
    valid_run_range = model_class.valid_run_range
    if run_no < valid_run_range[0] or run_no > valid_run_range[1]:
        raise ValueError("Requested run %d out of range (%d,%d)"%(run_no,valid_run_range[0],valid_run_range[1]))

    return _MakeAtariModel(class_map[algo],name,environment,model_path,run_no,algo,log_path,data_path)

if __name__=='__main__':
    #easy!
    Zaxxon_A2C = MakeAtariModel('rainbow','SeaquestNoFrameskip-v4',2,tag="final",local=False)
    model = Zaxxon_A2C()
    model.load_graphdef()
    model.import_graph()
    print("Done")
