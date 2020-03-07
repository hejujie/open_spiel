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
"""Tests for open_spiel.python.algorithms.alpha_zero.model."""

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v1 as tf

from open_spiel.python.algorithms.mu_zero import model as model_lib
import pyspiel

tf.enable_eager_execution()

solved = {}


def solve_game(state):
  state_str = str(state)
  if state_str in solved:
    return solved[state_str].value
  if state.is_terminal():
    return state.returns()[0]

  max_player = state.current_player() == 0
  obs = state.observation_tensor()
  act_mask = np.array(state.legal_actions_mask())
  values = np.full(act_mask.shape, -2 if max_player else 2)
  for action in state.legal_actions():
    values[action] = solve_game(state.child(action))
  value = values.max() if max_player else values.min()
  best_actions = np.where((values == value) & act_mask)
  policy = np.zeros_like(act_mask)
  policy[best_actions[0][0]] = 1  # Choose the first for a deterministic policy.
  solved[state_str] = model_lib.TrainInput(obs, act_mask, policy, value)
  return value


model_types = ["resnet", "mlp"]


def build_model(game, model_type):
  num_actions = game.num_distinct_actions()
  observation_shape = game.observation_tensor_shape()

  if model_type == "resnet":
    net = model_lib.keras_resnet(
        observation_shape, num_actions, num_residual_blocks=2, num_filters=64,
        value_head_hidden_size=64, data_format="channels_first")
  elif model_type == "mlp":
    net = model_lib.keras_mlp(
        observation_shape, num_actions, num_layers=2, num_hidden=64)
  else:
    raise ValueError("Invalid model_type: {}".format(model_type))

  return model_lib.Model(
      net, l2_regularization=1e-4, learning_rate=0.01, device="cpu")


class ModelTest(parameterized.TestCase):

  @parameterized.parameters(model_types)
  def test_model_learns_simple(self, model_type):
    game = pyspiel.load_game("tic_tac_toe")
    model = build_model(game, model_type)
    print("Num variables:", model.num_trainable_variables)
    model.print_trainable_variables()

    train_inputs = []
    state = game.new_initial_state()
    while not state.is_terminal():
      obs = state.observation_tensor()
      act_mask = state.legal_actions_mask()
      action = state.legal_actions()[0]
      policy = np.zeros(len(act_mask), dtype=float)
      policy[action] = 1
      train_inputs.append(model_lib.TrainInput(obs, act_mask, policy, value=1))
      state.apply_action(action)
      value, policy = model.inference([obs], [act_mask])
      self.assertLen(policy, 1)
      self.assertLen(value, 1)
      self.assertLen(policy[0], game.num_distinct_actions())
      self.assertLen(value[0], 1)

    losses = []
    for i in range(1000):
      loss = model.update(train_inputs)
      print(i, loss)
      losses.append(loss)
      if loss.policy < 0.05 and loss.value < 0.05:
        break

    self.assertGreater(losses[0].total, losses[-1].total)
    self.assertGreater(losses[0].policy, losses[-1].policy)
    self.assertGreater(losses[0].value, losses[-1].value)
    self.assertLess(losses[-1].value, 0.05)
    self.assertLess(losses[-1].policy, 0.05)

  @parameterized.parameters(model_types)
  def test_model_learns_optimal(self, model_type):
    game = pyspiel.load_game("tic_tac_toe")
    solve_game(game.new_initial_state())

    model = build_model(game, model_type)
    print("Num variables:", model.num_trainable_variables)
    model.print_trainable_variables()

    train_inputs = list(solved.values())
    print("states:", len(train_inputs))
    losses = []
    for i in range(2000):
      loss = model.update(train_inputs)
      print(i, loss)
      losses.append(loss)
      if loss.policy < 0.1 and loss.value < 0.1:
        break

    self.assertGreater(losses[0].policy, losses[-1].policy)
    self.assertGreater(losses[0].value, losses[-1].value)
    self.assertGreater(losses[0].total, losses[-1].total)
    self.assertLess(losses[-1].value, 0.1)
    self.assertLess(losses[-1].policy, 0.1)


if __name__ == "__main__":
  absltest.main()
