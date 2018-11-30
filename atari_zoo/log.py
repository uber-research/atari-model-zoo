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

import numpy as np
import pandas as pd
import json
import atari_zoo
from atari_zoo.utils import load_json_from_url

#TODO: Potentially refactor into class structure

"""
Helper function to parse checkpoint log to expose available 'tagged' checkpoints and their information
"""
def parse_checkpoint_info(json_data):
    ckt_points = {}
    for entry in json_data:
        tag = entry['criteria']
        if entry['best_checkpoint']!=None:
            ckt_points[tag]=entry['best_checkpoint']
    return ckt_points

"""
Load checkpoint json file from path, where
path can be either a local address or a web
address
"""
def load_checkpoint_info(path):
        fname = path+".json"

        if(fname.find('http')!=-1):
            return load_json_from_url(fname)
        else:
            return json.load(open(fname))

"""
Helper function to transform json log format to pandas data frames for plotting
"""
def get_dataframe_from_training_log(_data=None,_file=None,algo='default',run=1):
    if(not _data):
        log = json.load(_file)
    else:
        log = _data
    
    assert len(log)>0
    
    column_names = log[0].keys()
    data_dict = {}

    for column in column_names:
        data_dict[column] = []
            
    for entry in log:
        for key in column_names:
            data_dict[key].append(entry[key])
            
    data_dict['algo'] = [algo] * len(log)
    data_dict['run'] = [run] * len(log)
            
    df = pd.DataFrame(data_dict)
    
    if 'initial' in column_names:
        df = df[df['initial']==0]
        
    df = df.sort_values(by=['time'])

    """
    clean-up stage: some Ape-X runs were restarted from checkpoints
    due to a network outage. which creates a big timedelta that needs 
    to be cleaned up: i.e. 
    """
    clean=False
    threshold = 60*60*3 #assume >3 hour gap means restart

    while not clean:
        clean=True

        time_diffs = np.diff(df['time'])
        if np.max(time_diffs)>threshold:
            clean=False
            idx = np.argmax(time_diffs)
            amt = np.max(time_diffs)

            #TODO: use loc instead (don't operate on copy)
            #df['time'][idx+1:]-=amt
            df.loc[idx+1:,('time')]-=amt
    
    return df

"""
Helper function to gather logs for runs of a particular algo/game combo
"""
def gather_logs_across_runs(algo,game,runs,local=False):
    results = []

    for run in runs:
        k= atari_zoo.MakeAtariModel(algo,game,run,local=local)()
        log = k.get_log()
        results.append(get_dataframe_from_training_log(_data=log,algo=algo,run=run))

    df = pd.concat(results)
    return df


"""
Helper function to gather logs across algorithms for a particular game
"""
def gather_logs_across_algos(algos,game,local=False):
    results = []

    for algo in algos:
        results.append(gather_logs_across_runs(algo,game,range(1,atari_zoo.run_cnt[algo]+1),local=local))

    df = pd.concat(results)

    return df


if __name__=='__main__':
    import seaborn as sns
    from pylab import *
    algo = "apex"
    game = "AmidarNoFrameskip-v4"
    
    """
    apex_df = gather_logs_across_runs("apex",game,range(1,6),local=True)
    ga_df = gather_logs_across_runs("ga",game,range(1,6),local=True)
    a2c_df = gather_logs_across_runs("a2c",game,range(1,4),local=True)
    es_df = gather_logs_across_runs("es",game,range(1,4),local=True)

    df = pd.concat((apex_df,ga_df,a2c_df,es_df))
    """

    df = gather_logs_across_algos(['apex','ga','a2c','es'],game,local=True)

    sns.lineplot(x="time", y="score",
                              style="run", hue='algo',
                              data=df)

    show()
