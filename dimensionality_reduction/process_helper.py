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

from collections import OrderedDict
import numpy as np
from sklearn import decomposition, manifold

def assemble_data(data, ext=".", dict_key='ram'):
    """Assemble hi-D BCs from all generations"""


    #1 reconstruct the data directories
    X, npz_files, npz_dims = [], [], []
    raw_data = {}

    glo_max_score, glo_min_score = None, None

    algos_data = OrderedDict(data['algos'])
    for algo in algos_data.keys():
        for exp_idx in algos_data[algo]:
            path_to_npz = (data['path'] + '/' + algo + '/' + data['game'] + '/model'
                + str(exp_idx) + '_final_rollout'+ext+'npz')
            npz_files.append(path_to_npz)
            #check if path exist
            try:
                npz_data = np.load(path_to_npz)
            except:
                raise IOError("Invalid path: {}".format(path_to_npz))

            if algo not in raw_data.keys():
                raw_data[algo] = {}

            raw_data[algo][str(exp_idx)] = npz_data
            print(npz_data[dict_key].shape)
            print(type(npz_data[dict_key]))

            npz_dims.append(npz_data[dict_key].shape[0])
            X.append(npz_data[dict_key])

            if 'score' in npz_data.keys():
                loc_max, loc_min = max(npz_data['score']), min(npz_data['score'])

                if glo_max_score == None or loc_max > glo_max_score:
                    glo_max_score = loc_max

                if glo_min_score == None or loc_min < glo_min_score:
                    glo_min_score = loc_min
                print(loc_max, loc_min)

    return X, npz_files, npz_dims, raw_data, glo_max_score, glo_min_score

def reduce_dim(X, method):
    print(method)

    print("Reducing ...")

    perplexity = 30
    n_iter = 1000
    pca_dim = 50
    if 'tsne' in method['name']:
        if 'perplexity' in method.keys():
            perplexity = method['perplexity']
        if 'n_iter' in method.keys():
            n_iter = method['n_iter']
        if 'pca_dim' in method.keys():
            pca_dim = method['pca_dim']

    print(perplexity, n_iter, pca_dim)

    if method['name'] == 'tsne':
        print('running tsne')
        X_r = manifold.TSNE(n_components=2, perplexity=perplexity,
                            verbose=2, random_state=0, n_iter=n_iter).fit_transform(X)
    elif method['name'] == 'pca':
        print('running pca')
        X_r = decomposition.PCA(n_components=2).fit_transform(X)
    elif method['name'] == 'pca_tsne':
        print('running pca')
        X_pca = decomposition.PCA(n_components=pca_dim).fit_transform(X)
        print('running tsne')
        X_r = manifold.TSNE(n_components=2, perplexity=perplexity,
                            verbose=2, random_state=0, n_iter=n_iter).fit_transform(X_pca)
    elif method['name'] == 'debug':
        print('running debug')
        nrow, ncol = X.shape
        idx_last_x, idx_last_y = int(ncol / 2 - 1), -1
        X_r = np.hstack((X[:, idx_last_x].reshape(nrow, 1), X[:, idx_last_y].reshape(nrow, 1)))
    else:
        raise NotImplementedError

    print('Reduction Completed! X.shape={} X_r.shape={}'.format(X.shape, X_r.shape))
    return X_r


def disassemble(X, files, dims, dr_method, dict_key='ram'):
    print(dr_method)
    dict_key_2d = dict_key+"_2d"
    X_splitted = np.split(X, np.cumsum(dims)[:-1])
    for x_2d, file in zip(X_splitted, files):
        npz_name = file.split('.')[0] + '.' + dr_method + '.' + dict_key_2d
        print(x_2d.shape, npz_name)
        vals_to_save = {dict_key_2d:x_2d}

        np.savez_compressed(npz_name, **vals_to_save)


    if dr_method == "debug":
        for file in files:
            data = np.load(file)
            npz_name = file.split('.')[0] + '.' + dr_method + '.' + dict_key_2d + '.npz'
            data_2d = np.load(npz_name)

            print(data[dict_key].shape)
            print(data_2d[dict_key_2d].shape)

            assert data[dict_key].shape[0] == data_2d[dict_key_2d].shape[0]
            assert np.array_equal(data[dict_key][:, int(data[dict_key].shape[1] / 2 - 1) ],  data_2d[dict_key_2d][:, 0])
            assert np.array_equal(data[dict_key][:, -1], data_2d[dict_key_2d][:, 1])




