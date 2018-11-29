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
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from tensorflow import keras


algos = ['a2c','es','ga','apex','rainbow','dqn']

algos_2 = algos[:6]
env_name = 'SeaquestNoFrameskip-v4'
# env_name = 'ZaxxonNoFrameskip-v4'
root_dir = '/rl_zoo/data/'
rollout_path_template = root_dir + 'rlzoo/{}/{}/model{}_rollout.npz'
split_by_rollout = False
n_last_frames = 1
num_models = 3

train_frames, train_labels, test_frames, test_labels = None, None, None, None

if split_by_rollout:
    idx = np.arange(num_models)
    np.random.shuffle(idx)
    train_idx, test_idx = idx[:-1], idx[-1:]
    for algo in algos_2:
        for model_num in range(num_models):
            rollout = np.load(rollout_path_template.format(algo, env_name, model_num + 1))
            observations = rollout['observations'][:, :, :, :n_last_frames]
            labels_ = np.empty(observations.shape[0])
            labels_.fill(algos_2.index(algo))
            if model_num in train_idx:
                if train_frames is None:
                    train_frames = observations
                    train_labels = labels_
                else:
                    train_frames = np.append(train_frames, observations, axis=0)
                    train_labels = np.append(train_labels, labels_, axis=0)
            elif model_num in test_idx:
                if test_frames is None:
                    test_frames = observations
                    test_labels = labels_
                else:
                    test_frames = np.append(test_frames, observations, axis=0)
                    test_labels = np.append(test_labels, labels_, axis=0)
else:
    idx = np.arange(num_models)
    np.random.shuffle(idx)
    train_idx, test_idx = idx[:-1], idx[-1:]
    frames, labels = None, None
    for algo in algos_2:
        for model_num in range(num_models):
            rollout = np.load(rollout_path_template.format(algo, env_name, model_num + 1))
            observations = rollout['observations'][:, :, :, :n_last_frames]
            labels_ = np.empty(observations.shape[0])
            labels_.fill(algos_2.index(algo))
            if model_num in train_idx:
                if frames is None:
                    frames = observations
                    labels = labels_
                else:
                    frames = np.append(frames, observations, axis=0)
                    labels = np.append(labels, labels_, axis=0)
    indices = np.random.permutation(frames.shape[0])
    split_threshold = int(frames.shape[0] * 0.8)
    train_idx, test_idx = indices[:split_threshold], indices[split_threshold:]
    train_frames, train_labels = frames[train_idx, :], labels[train_idx]
    test_frames, test_labels = frames[test_idx, :], labels[test_idx]

print('train shape', train_frames.shape, train_labels.shape)
print('test shape', test_frames.shape, test_labels.shape)

frame_size = train_frames.shape[1]
kernel_size = 3
model = keras.Sequential([
    keras.layers.Conv2D(16, kernel_size, activation=tf.nn.relu, input_shape=(frame_size, frame_size, n_last_frames)),
    keras.layers.Conv2D(32, kernel_size, activation=tf.nn.relu),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(len(algos_2), activation=tf.nn.softmax)
])
model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_frames, train_labels, epochs=1, callbacks=[keras.callbacks.EarlyStopping()], validation_split=0.1)

# eval
test_loss, test_acc = model.evaluate(test_frames, test_labels)
print('Overall test accuracy:', test_acc)
pred_labels = model.predict_classes(test_frames)
cnf_matrix = confusion_matrix(test_labels, pred_labels)
print('confusion matrix')
print(cnf_matrix)
print(classification_report(test_labels, pred_labels, target_names=algos_2))
