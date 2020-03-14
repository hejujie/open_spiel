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


class TrainInput(collections.namedtuple(
        "TrainInput", "observation actions target_value, target_reward, target_policy")):
    """Inputs for training the Model."""

    @staticmethod
    def stack(train_inputs):
        observation, actions, target_vs, target_rs, target_ps = zip(
            *train_inputs)
        actions = list(zip(actions))
        # print("before", target_vs)
        target_vs = list(zip(*list(target_vs)))
        target_rs = list(zip(*list(target_rs)))
        target_ps = list(zip(*list(target_ps)))
        result = TrainInput(
            np.array(observation, dtype=np.float32),
            np.expand_dims(np.array(actions), -1),
            np.expand_dims(np.array(target_vs), -1),
            np.expand_dims(np.array(target_rs), -1),
            np.array(target_ps))
        # print("target out", result.targets)
        # print("obs", result.observation.shape)
        # print("action out", result.actions)
        # print("target_vs", result.target_value.shape)
        # print("target_rs", result.target_reward.shape)
        # print("target_ps", result.target_policy.shape)

        return result


class Losses(collections.namedtuple("Losses", "policy value reward l2")):
    """Losses from a training step."""

    @property
    def total(self):
        return self.policy + self.value + self.reward + self.l2

    def __str__(self):
        return ("Losses(total: {:.3f}, policy: {:.3f}, value: {:.3f}, reward: {:.3f}, "
                "l2: {:.3f})").format(self.total, self.policy, self.value, self.reward, self.l2)

    def __add__(self, other):
        return Losses(self.policy + other.policy,
                      self.value + other.value,
                      self.reward + other.reward,
                      self.l2 + other.l2)

    def __truediv__(self, n):
        return Losses(self.policy / n, self.value / n, self.reward / n, self.l2 / n)


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
        self.initial_inference = self.initial_inference_process()
        self.recurrent_inference = self.recurrent_inference_process()

    def representation_network(self):
        raise NotImplementedError

    def dynamic_network(self):
        raise NotImplementedError

    def prediction_network(self):
        raise NotImplementedError

    def initial_inference_process(self):
        input_size = int(np.prod(self.observation_shape))
        inputs = tf.keras.Input(
            shape=input_size, dtype="float32", name="input")
        hidden_state = self.representation_network(inputs)
        value, policy_logits = self.prediction_network(hidden_state)
        return tf.keras.Model(inputs=[inputs],
                              outputs=[value, policy_logits, hidden_state])

    def recurrent_inference_process(self):
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


def scalar_loss(prediction, target):
    return tf.keras.losses.MSE(target, prediction)


def scale_gradient(loss, factor):
    return (1 - factor) * tf.stop_gradient(loss) + factor * loss


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
            value, policy, hidden_state = self._muzero_model.initial_inference([
                np.array(obs, dtype=np.float32)])
            return value, value*0, policy, hidden_state

    def recurrent_inference(self, hidden_state):
        with self._device:
            return self._muzero_model.recurrent_inference([
                hidden_state])

    def update(self, train_inputs: Sequence[TrainInput]):
        batch = TrainInput.stack(train_inputs)

        policy_loss = 0
        value_loss = 0
        reward_loss = 0
        l2_loss = 0
        with self._device:
            with tf.GradientTape() as tape:
                observation, actions = batch.observation, batch.actions
                value, policy_logits, next_hidden = self._muzero_model.initial_inference(
                    [observation], training=True)
                predictions = [(1.0, value, value*0, policy_logits)]
                for action in actions:
                    value, reward, policy_logits, next_hidden = self._muzero_model.recurrent_inference(
                        [next_hidden], training=True)
                    predictions.append(
                        (1/len(actions), value, reward, policy_logits))
                    next_hidden = scale_gradient(next_hidden, 0.5)

                for prediction, *target in zip(predictions, batch.target_value,
                                               batch.target_reward, batch.target_policy):
                    gradient_scale, value, reward, policy_logits = prediction
                    target_value, target_reward, target_policy = target
                    value_loss += tf.reduce_mean(scale_gradient(
                        scalar_loss(value, target_value), gradient_scale))
                    reward_loss += tf.reduce_mean(scale_gradient(
                        scalar_loss(reward, target_reward), gradient_scale))
                    policy_loss += tf.reduce_mean(scale_gradient(tf.nn.softmax_cross_entropy_with_logits(
                        logits=policy_logits, labels=target_policy), gradient_scale))

                l2_loss += tf.add_n([self._l2_regularization * tf.nn.l2_loss(var)
                                     for var in self._muzero_model.initial_inference.trainable_variables
                                     if "/bias:" not in var.name])
                l2_loss += tf.add_n([self._l2_regularization * tf.nn.l2_loss(var)
                                     for var in self._muzero_model.recurrent_inference.trainable_variables
                                     if "/bias:" not in var.name])

                loss = value_loss + reward_loss + policy_loss + l2_loss

            trainable_variable = self.get_trainable_variable()
            grads = tape.gradient(loss, trainable_variable)

            self._optimizer.apply_gradients(
                zip(grads, trainable_variable),
                global_step=tf.train.get_or_create_global_step())
        return Losses(policy=float(policy_loss), value=float(value_loss),
                      reward=float(reward_loss), l2=float(l2_loss))

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

    def get_trainable_variable(self):
        variable = (self._muzero_model.initial_inference.trainable_variables +
                    self._muzero_model.recurrent_inference.trainable_variables)
        return variable


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
