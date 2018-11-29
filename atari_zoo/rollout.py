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

from pdb import set_trace as bb
import argparse
#from utils import *
#from models import *
#from ga_vis import create_ga_model 
import pickle

import lucid
from lucid.modelzoo.vision_base import Model
from lucid.misc.io import show
import lucid.optvis.objectives as objectives
import lucid.optvis.param as param
import lucid.optvis.transform as transform
import lucid.optvis.render as render
import tensorflow as tf

from atari_zoo import MakeAtariModel
from lucid.optvis.render import import_model
import gym
import atari_zoo.atari_wrappers as atari_wrappers
import numpy as np
import random
from atari_zoo.dopamine_preprocessing import AtariPreprocessing as DopamineAtariPreprocessing	
from atari_zoo.atari_wrappers import FireResetEnv, NoopResetEnv, MaxAndSkipEnv,WarpFrameTF,FrameStack,ScaledFloatFrame 

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def generate_rollout(model,args=None,action_noise=0.0,parameter_noise=0.0,observation_noise=0.0,test_eps=1,max_frames=2500,min_frames=2500,output='',sticky_action_prob=0.0,render=False,cpu=False,streamline=False,verbose=False):

    if args==None:
        arg_dict = {'parameter_noise':parameter_noise,
                    'observation_noise':observation_noise,
                    'test_eps':test_eps,
                    'max_frames':max_frames,
                    'min_frames':min_frames,
                    'output':output,
                    'sticky_action_prob':sticky_action_prob,
                    'render':render,
                    'streamline':streamline,
                    'action_noise':action_noise,
                    'verbose':verbose
                    }
        args = dotdict(arg_dict)
    
    #from machado
    sticky_action_prob = args.sticky_action_prob
    
    m = model 

    preprocessing = m.preprocess_style

    m.load_graphdef()

    #modify graphdef with gaussian noise
    if args.parameter_noise > 0.0:
        perturb_count = 0
        layer_names = [z['name'] for z in m.weights]

        #search for known-named nodes
        for node in m.graph_def.node:
            if node.name in layer_names:
                #black magic
                tensor = node.attr.get('value').tensor
                array = np.frombuffer(tensor.tensor_content,np.float32).copy()
                array += np.random.normal(0,args.parameter_noise,array.shape)
                tensor.tensor_content = array.tobytes()
                perturb_count+=1

        #print(perturb_count)
        #[n.name for n in m.graph_def.node if n.name.find("conv")!=-1]
        #bb()

        #should hit 3 conv layers
        assert perturb_count == 3

    dev_cnt = 1
    if args.cpu:
        dev_cnt = 0
    #for rollouts maybe don't use GPU?
    config = tf.ConfigProto(
            device_count = {'GPU': dev_cnt}
        )
    config.gpu_options.allow_growth=True

    with tf.Graph().as_default() as graph, tf.Session(config=config) as sess:
 
        if preprocessing == 'dopamine': #dopamine-style preprocessing
            env = gym.make(m.environment)
            if hasattr(env,'unwrapped'):
                env = env.unwrapped
            env = DopamineAtariPreprocessing(env)
            env = FrameStack(env, 4)
            env = ScaledFloatFrame(env,scale=1.0/255.0)
        elif preprocessing == 'np': #use numpy preprocessing
            env = gym.make(m.environment)
            env = atari_wrappers.wrap_deepmind(env, episode_life=False,preproc='np')
        else:  #use tensorflow preprocessing
            env = gym.make(m.environment)
            env = atari_wrappers.wrap_deepmind(env, episode_life=False,preproc='tf')

        nA = env.action_space.n
        X_t = tf.placeholder(tf.float32, [None] + list(env.observation_space.shape))

        T = import_model(m,X_t,X_t)
        action_sample = m.get_action(T)

        #get intermediate level representations
        activations = [T(layer['name']) for layer in m.layers]
        high_level_rep = activations[-2] #not output layer, but layer before

        sample_observations = []
        sample_frames = []
        sample_ram = []
        sample_representation = []
        sample_score = []

        obs = env.reset()

        ep_count = 0
        rewards = []; ep_rew = 0.
        frame_count = 0
    
        prev_action = None

        # Evaluate policy over test_eps episodes
        while ep_count < args.test_eps or frame_count<=args.min_frames:
            if args.render:
                env.render()

            #potentially add observation noise
            if args.observation_noise>0.0:
                obs += np.random.normal(0,args.observation_noise,obs.shape)


            train_dict = {X_t:obs[None]}
            if streamline:
                results = sess.run([action_sample], feed_dict=train_dict)
                #grab action
                act = results[0]
            else:
                results = sess.run([action_sample,high_level_rep], feed_dict=train_dict)

                #grab action
                act = results[0]

                #get high-level representation
                representation = results[1][0]

            if not streamline:
                frame = env.render(mode='rgb_array')
                sample_frames.append(np.array(frame,dtype=np.uint8))
                sample_ram.append(env.unwrapped._get_ram())
                sample_representation.append(representation)
                sample_observations.append(np.array(obs))

            sample_score.append(ep_rew)

            if args.action_noise >=0:
                if random.random() < args.action_noise:
                    act = random.randint(0,nA-1)

            if prev_action != None and random.random() < sticky_action_prob:
                act = prev_action

            prev_action = act

            obs, rew, done, info = env.step(np.squeeze(act))

            ep_rew += rew
            frame_count+=1

            if frame_count >= args.max_frames:
                done=True

            if done:
                obs = env.reset()
                ep_count += 1
                rewards.append(ep_rew)
                ep_rew = 0.

        if args.verbose:
            print("Avg. Episode Reward: ", np.mean(rewards))
            print("rewards:",rewards)
            print("frames:",frame_count)

        results = {'observations':sample_observations,'frames':sample_frames,'ram':sample_ram,'representation':sample_representation,'score':sample_score,'ep_rewards':rewards}

        if args.output!='':
            np.savez_compressed(args.output + "_rollout",**results)

        return results 


#TODO wrap this as a function call, so you can do multiple rollouts
def main():
    """
    Rolls out a model in the atari emulator -- can render it to screen, and also
    can save out image and observation sequences.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--test_eps', help='number of test episodes', default=1, type=int)
    parser.add_argument('--algo', help='choose from [es, a2c, dqn]', type=str)
    parser.add_argument('--environment', type=str)
    parser.add_argument('--run_id',type=int, default=1)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--output', type=str, default="")
    parser.add_argument('--max_frames', type=int, default=1e8)
    parser.add_argument('--min_frames', type=int, default=0)
    parser.add_argument('--observation_noise', type=float, default=0.0)
    parser.add_argument('--parameter_noise', type=float, default=0.0)
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--streamline', action='store_true')
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--sticky_action_prob', type=float,default=0.0)
    parser.add_argument('--action_noise', type=float,default=0.0)
    parser.add_argument('--verbose', action="store_true")

    #from machado
    sticky_action_prob = 0.0

    args = parser.parse_args()
    
    m = MakeAtariModel(args.algo,args.environment,args.run_id,tag=args.tag,local=args.local)()
   
    results = generate_rollout(model=m,args=args)

    exit()

if __name__=="__main__":
    #generate_rollout(blah="blah2")
    main()
