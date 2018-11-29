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

import os
import glob
import sys
from pdb import set_trace as bb


def module_path():
    encoding = sys.getfilesystemencoding()
    return os.path.dirname(__file__)

path = module_path()

dopamine_game_list = open(os.path.join(path,"game_lists/dopamine_game_list")).read().split("\n")[:-1]
es_apex_game_list = open(os.path.join(path,"game_lists/apex_game_list")).read().split("\n")[:-1]
a2c_game_list = open(os.path.join(path,"game_lists/a2c_game_list")).read().split("\n")[:-1]

#align game lists by taking out these games
blacklist = ['AirRaid','Carnival','ElevatorAction','JourneyEscape','Pooyan']

dopamine_game_list = [k for k in dopamine_game_list if k not in blacklist]
a2c_game_list = [k for k in a2c_game_list if (k[:k.find("NoFrameskip-v4")]) not in blacklist]


def grab_list(mode):
        if mode=='a2c' or mode=='canonical':
            return a2c_game_list
        if mode=='es' or mode=='apex':
            return es_apex_game_list
        if mode=='dopamine':
            return dopamine_game_list

def translate_game_name(inp_name,inp_mode,out_mode):
    inp_list = None
    out_list = None

    inp_list = grab_list(inp_mode)
    out_list = grab_list(out_mode)

    inp_idx = inp_list.index(inp_name)

    return out_list[inp_idx]


if __name__=='__main__':

    for k in range(len(dopamine_game_list)):
        print(dopamine_game_list[k],es_apex_game_list[k],a2c_game_list[k])

    print(translate_game_name('ice_hockey','apex','canonical'))

    print(a2c_game_list)
