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

import matplotlib.pyplot as plt
from process_helper import assemble_data
from matplotlib.widgets import CheckButtons
import numpy as np
from colour import Color
from mpldatacursor import datacursor

artist2data = {}
label2artist = {}

COLORS = [
    (Color('#f9d9d9'), Color('#d61515')), # red
    (Color('#d9ddfb'), Color('#0b1667')), # blue
    (Color('#9aecb8'), Color('#045c24')), # green
    (Color('#ffbef9'), Color('#ce00bb')), # pink
    (Color('#ffb27e'), Color('#fb6500')), # orange
    (Color('#d0d0d0'), Color('#000000')), # black
    (Color('#beffcf'), Color('#33FF66')), # lime green
    (Color('#f2d6b9'), Color('#996633')), # brown
    (Color('#d5b2ec'), Color('#9900FF')), # purple
    (Color('#baffff'), Color('#009999')), # teel
]

numBins = 255
COLOR_HEX_LISTS = []
for color in COLORS:
    color_gradient = color[0].range_to(color[1], numBins)
    hex_list = [c.get_hex_l() for c in color_gradient]
    COLOR_HEX_LISTS.append(hex_list)

def color_index(fitness, minfit, maxfit):
    cind = (fitness - minfit)/(maxfit - minfit) * numBins
    cind = int(cind)
    if cind >= numBins:
        cind = numBins-1
    elif cind < 0:
        cind = 0

    return cind

def color_list(color_idx, scores, global_limit=None):
    this_color = COLOR_HEX_LISTS[color_idx]
    if global_limit:
        loc_max, loc_min = global_limit
    else:
        loc_max, loc_min = max(scores), min(scores)

    hex_list = []
    for s in scores:
        hex_list.append(this_color[color_index(s, loc_min, loc_max)])
    return hex_list

def checkb_click(label):

    print("clicked", label)
    artist = label2artist[label]
    artist.set_visible(not artist.get_visible())

    visible = artist.get_visible()
    if visible:
        artist.set_picker(5)
    else:
        artist.set_picker(None)
    plt.draw()


class figure_control:
    def __init__(self, config, global_max_min):
        self.config = config
        print(config)

        _, _, _, self.raw_data, self.glo_max_score, self.glo_min_score = assemble_data(config['data'])

        Xs, npz_files, npz_dims, _, _, _ = assemble_data(config['data'],
            ext='.' + config['method']['name'] + '.'+config['data']['key'] + '.', dict_key=config['data']['key'])


        print(self.glo_max_score, self.glo_min_score)
        print(npz_dims, npz_files)
        self.fig = plt.figure(config['data']['key'].upper(), figsize=(14,7))
        self.ax = self.fig.add_subplot(111)
        plt.subplots_adjust(left  = 0.5, right = 0.9)
        self.fig.canvas.mpl_connect('pick_event', self.onpick)


        checkb_labels = []
        idx = 0
        list_of_artists = []
        labels_cursor = {}
        for data, file in zip(Xs, npz_files):
            print(data.shape)
            algo = file.split('/')[-3]
            rollout = int(file.split('/')[-1].split('_')[0][-1])
            scores = self.raw_data[algo][str(rollout)]['score']
            if global_max_min:
                global_limit = (self.glo_max_score, self.glo_min_score)
            else:
                global_limit = None
            hex_list = color_list(idx%len(COLORS), scores, global_limit=global_limit )
            artist = self.ax.scatter(data[:, 0], data[:, 1], c=hex_list, visible=False)

            list_of_artists.append(artist)
            labels_cursor[artist]=[]
            for i in range(data.shape[0]):
                labels_cursor[artist].append(algo.upper() + " r" + str(rollout)
                                             + " s"+str(i) + ": "+str(scores[i]))

            #artist.set_visible(True)
            #artist.set_picker(5)
            artist2data[artist] = (algo, rollout)

            label = algo.upper() + " rollout#" + str(rollout)
            label2artist[label] = artist
            checkb_labels.append(label)
            idx += 1

        formatter = lambda **kwargs: ', '.join(kwargs['point_label'])
        self.dc = datacursor(list_of_artists, hover=True, formatter=formatter,
                             point_labels=labels_cursor)


        checkb_ax = self.fig.add_axes([0.1, 0.1, 0.4, 0.8])
        checkb_ax.axis('off')
        self.checkb = CheckButtons(checkb_ax,
                                    checkb_labels, [False]*len(checkb_labels))
        self.checkb.on_clicked(checkb_click)


    def pop_frame(self, algo, rollout, index):
        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_subplot(111)
        fig.suptitle(algo + " rollout #" + str(rollout)
                     + " step " + str(index)
                     + " score "+str(self.raw_data[algo][str(rollout)]['score'][index]))
        print(algo, rollout, index)
        ax.imshow(self.raw_data[algo][str(rollout)]['frames'][index])
        fig.show()


    def onpick(self, event):
        print(hasattr(event, 'ind'))
        print(hasattr(event, 'artist'))
        if hasattr(event, 'ind') and hasattr(event, 'artist'):
            print("picked", event.ind)
            index = event.ind[-1]
            algo, rollout = artist2data[event.artist]
            print(index)
            print(artist2data[event.artist])
            print(event.artist.get_offsets()[index])
            self.pop_frame(algo, rollout, index)
