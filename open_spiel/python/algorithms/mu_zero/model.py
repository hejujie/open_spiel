# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from typing import Sequence

import numpy as np
import tensorflow.compat.v1 as tf


class BaseMuZeroModel(object):
    def __init__(self):
        pass

    def representation_network(self):
        raise NotImplementedError

    def dynamic_network(self):
        raise NotImplementedError

    def prediction_network(self):
        raise NotImplementedError

    def initial_inference(self) -> tf.keras.Model:
        pass

    def recurrent_inference(self) -> tf.keras.Model:
        pass

    def reward_transform(self):
        pass

    def value_transform(self):
        pass


class MuZeroGoModel(BaseMuZeroModel):
    pass


class MuZeroAtariModel(BaseMuZeroModel):
    pass


class Model(object):

    def __init__(self, muzero_model: BaseMuZeroModel, l2_regularization, learning_rate, device):

        if device == "gpu":
            if not tf.test.is_gpu_available():
                raise ValueError("GPU support is unavailable.")
            self._device = tf.device("gpu:0")
        elif device == "cpu":
            self._device = tf.device("cpu:0")
        else:
            self._device = device
        self._muzero_model = muzero_model
        self._optimizer = tf.train.AdamOptimizer(learning_rate)
        self._l2_regularization = l2_regularization

    def initial_inference(self):
        _ = self._muzero_model.initial_inference()
        pass

    def recurrent_inference(self):
        _ = self._muzero_model.recurrent_inference()
        pass

    def compute_loss(self):
        pass

    def update(self):
        # get batch data
        # get loss for difference step
        # grads, apply_grads
        pass

    def conditional_state(self):
        # get conditional_state when get loss
        pass


def cascade(x, fns):
    for fn in fns:
        x = fn(x)
    return x


def keras_resnet(inputs,
                 num_residual_blocks=16,
                 num_filters=256,
                 kernel_size=3,
                 activation="relu"):

    def residual_layer(inputs, num_filters, kernel_size):
        return cascade(inputs, [
            tf.keras.layers.Conv2D(num_filters, kernel_size, padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(activation),
            tf.keras.layers.Conv2D(num_filters, kernel_size, padding="same"),
            tf.keras.layers.BatchNormalization(axis=-1),
            lambda x: tf.keras.layers.add([x, inputs]),
            tf.keras.layers.Activation(activation),
        ])

    def resnet_body(inputs, num_filters, kernel_size):
        x = cascade(inputs, [
            tf.keras.layers.Conv2D(num_filters, kernel_size, padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(activation),
        ])
        for _ in range(num_residual_blocks):
            x = residual_layer(x, num_filters, kernel_size)
        return x

    return resnet_body(inputs, num_filters, kernel_size)


def keras_mlp(inputs,
              num_layers=2,
              num_hiddens=128,
              activation="relu"):

    output = inputs
    for _ in range(num_layers):
        output = tf.keras.layers.Dense(
            num_hidden, activation=activation)(output)
    return output
