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


class MuZeroBaseModel(object):
    def __init__(self,
                 observation_shape,
                 num_actions,
                 num_layers=2,
                 num_hiddens=128,
                 activation="relu"):
        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.num_layers = num_layers
        self.num_hiddens = num_hiddens
        self.activation = activation

    def representation_network(self):
        raise NotImplementedError

    def dynamic_network(self):
        raise NotImplementedError

    def prediction_network(self):
        raise NotImplementedError

    @property
    def initial_inference(self):
        input_size = int(np.prod(self.observation_shape))
        inputs = tf.keras.Input(
            shape=input_size, dtype="float32", name="input")
        hidden_state = self.representation_network(inputs)
        value, policy_logits = self.prediction_network(hidden_state)
        return tf.keras.Model(inputs=[inputs],
                              outputs=[value, policy_logits, hidden_state])

    @property
    def recurrent_inference(self):
        hidden_state = tf.keras.Input(
            shape=self.num_hiddens, dtype="float32", name="hidden_state")
        next_hidden_state, reward = self.dynamic_network(hidden_state)
        value, policy_logits = self.prediction_network(hidden_state)
        return tf.keras.Model(inputs=[hidden_state],
                              outputs=[value, reward, policy_logits, next_hidden_state])

    def reward_transform(self):
        pass

    def value_transform(self):
        pass


class MuZeroGoModel(MuZeroBaseModel):
    pass


class MuZeroAtariModel(MuZeroBaseModel):
    pass


def keras_mlp(inputs,
              num_layers=2,
              num_hiddens=128,
              activation="relu",
              name=None):

    output = inputs
    for i in range(num_layers):
        output = tf.keras.layers.Dense(
            num_hiddens, activation=activation, name=name+'_'+str(i))(output)
    return output


class MuZeroMLPModel(MuZeroBaseModel):
    def __init__(self,
                 observation_shape,
                 num_actions,
                 num_layers=2,
                 num_hiddens=128,
                 activation="relu"):
        super(MuZeroMLPModel, self).__init__(observation_shape,
                                             num_actions, num_layers, num_hiddens, activation)

    def representation_network(self, inputs):
        initial_hidden_state = keras_mlp(
            inputs, self.num_layers, self.num_hiddens, name="representation")
        return initial_hidden_state

    def prediction_network(self, hidden_state):
        torso = keras_mlp(hidden_state, self.num_layers,
                          self.num_hiddens, name="prediction")

        policy_logits = tf.keras.layers.Dense(
            self.num_actions, name="policy")(torso)
        value = tf.keras.layers.Dense(
            1, activation="tanh", name="value")(torso)
        return value, policy_logits

    def dynamic_network(self, hidden_state):
        next_hidden_state = keras_mlp(
            hidden_state, self.num_layers, self.num_hiddens, name='dynamic')

        torso = keras_mlp(
            hidden_state, self.num_layers, self.num_hiddens, name='reward')
        reward = tf.keras.layers.Dense(
            1, activation="tanh", name='reward')(torso)
        return next_hidden_state, reward


class Model(object):

    def __init__(self, muzero_model: MuZeroBaseModel, l2_regularization, learning_rate, device):

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

    def initial_inference(self, obs):
        with self._device:
            return self._muzero_model.initial_inference([
                np.array(obs, dtype=np.float32)])

    def recurrent_inference(self, hidden_state):
        with self._device:
            return self._muzero_model.recurrent_inference([
                hidden_state])

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

    @property
    def num_trainable_variables(self):
        num_variables_initial = sum(np.prod(
            v.shape) for v in self._muzero_model.initial_inference.trainable_variables)
        num_variable_recurrent = sum(np.prod(
            v.shape) for v in self._muzero_model.recurrent_inference.trainable_variables)
        return num_variables_initial + num_variable_recurrent

    def print_trainable_variables(self):
        for v in self._muzero_model.initial_inference.trainable_variables:
            print("{}: {}".format(v.name, v.shape))
        for v in self._muzero_model.recurrent_inference.trainable_variables:
            print("{}: {}".format(v.name, v.shape))


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
