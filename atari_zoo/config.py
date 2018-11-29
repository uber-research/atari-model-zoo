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

def dopamine_url_formatter(base_url,agent,game,run,tag=None):
    game_proc = game.split("NoFrameskip")[0]
    return "gs://download-dopamine-rl/lucid/{agent}/{game}/{run}/graph_def.pb".format(agent=agent,game=game_proc,run=run)


#remote lookup table
datadir_remote_dict = {'apex':"https://dgqeqexrlnkvd.cloudfront.net/zoo/apex",
                         'es':"https://dgqeqexrlnkvd.cloudfront.net/zoo/es",
                         'ga':"https://dgqeqexrlnkvd.cloudfront.net/zoo/ga",
                         'a2c':"https://dgqeqexrlnkvd.cloudfront.net/zoo/a2c",
                         'rainbow':"https://dgqeqexrlnkvd.cloudfront.net/zoo/rainbow",
                       'dqn':"https://dgqeqexrlnkvd.cloudfront.net/zoo/rainbow"}

url_formatter_dict = {('rainbow','remote'):dopamine_url_formatter,('dqn','remote'):dopamine_url_formatter}


#local lookup table
datadir_local_dict = {'apex':"/space/rlzoo/apex",
                        'es':"/space/rlzoo/es",
                        'ga':"/space/rlzoo/ga",
                        'a2c':'/space/rlzoo/a2c',
                        'rainbow':'/space/rlzoo/rainbow',
                        'dqn':'/space/rlzoo/dqn'}

debug = True
