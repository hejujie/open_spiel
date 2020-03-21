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
"""Tests for google3.third_party.open_spiel.python.algorithms.alpha_zero.evaluator."""

from absl.testing import absltest
import numpy as np
import tensorflow.compat.v1 as tf

from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms.mu_zero import evaluator as evaluator_lib
from open_spiel.python.algorithms.mu_zero import model as model_lib
import pyspiel

tf.enable_eager_execution()


def build_model(game) -> model_lib.Model:
    num_actions = game.num_distinct_actions()
    observation_shape = game.observation_tensor_shape()
    net = model_lib.MuZeroMLPModel(observation_shape, num_actions)
    return model_lib.Model(
        net, l2_regularization=1e-4, learning_rate=0.001, device="cpu")


class EvaluatorTest(absltest.TestCase):

    def test_evaluator_initial_network_caching(self):
        game = pyspiel.load_game("tic_tac_toe")
        model = build_model(game)
        evaluator = evaluator_lib.MuZeroEvaluator(game, model)

        state = game.new_initial_state()

        value = evaluator.initial_evaluate(state)
        self.assertEqual(value[0], -value[1])
        value = value[0]

        value2 = evaluator.initial_evaluate(state)[0]
        self.assertEqual(value, value2)

        prior = evaluator.initial_prior(state)
        prior2 = evaluator.initial_prior(state)
        np.testing.assert_array_equal(prior, prior2)

        info = evaluator.initial_cache_info()
        self.assertEqual(info.misses, 1)
        self.assertEqual(info.hits, 3)

        # change model in evalutor
        evaluator._model = build_model(game)

        # Still equal due to not clearing the cache
        value3 = evaluator.initial_evaluate(state)[0]
        self.assertEqual(value, value3)

        info = evaluator.initial_cache_info()
        self.assertEqual(info.misses, 1)
        self.assertEqual(info.hits, 4)

        evaluator.clear_cache()

        info = evaluator.initial_cache_info()
        self.assertEqual(info.misses, 0)
        self.assertEqual(info.hits, 0)

        # Now they differ from before
        value4 = evaluator.initial_evaluate(state)[0]
        value5 = evaluator.initial_evaluate(state)[0]
        self.assertNotEqual(value, value4)
        self.assertEqual(value4, value5)

        info = evaluator.initial_cache_info()
        self.assertEqual(info.misses, 1)
        self.assertEqual(info.hits, 1)

        value6 = evaluator.initial_evaluate(game.new_initial_state())[0]
        self.assertEqual(value4, value6)

        info = evaluator.initial_cache_info()
        self.assertEqual(info.misses, 1)
        self.assertEqual(info.hits, 2)

    def test_evaluator_recurrent_network_caching(self):
        game = pyspiel.load_game("tic_tac_toe")
        model = build_model(game)
        evaluator = evaluator_lib.MuZeroEvaluator(game, model)

        state = game.new_initial_state()
        obs = state.observation_tensor()
        next_hidden = model.initial_inference([obs])[-1]
        next_hidden2 = model.initial_inference([obs])[-1]
        tf.debugging.assert_equal(next_hidden, next_hidden2)

        value = evaluator.recurrent_evaluate(next_hidden)
        self.assertEqual(value[0], -value[1])
        value = value[0]

        value2 = evaluator.recurrent_evaluate(next_hidden)[0]
        self.assertEqual(value, value2)

        prior = evaluator.recurrent_prior(next_hidden)
        prior2 = evaluator.recurrent_prior(next_hidden)
        np.testing.assert_array_equal(prior, prior2)

        info = evaluator.recurrent_cache_info()
        self.assertEqual(info.misses, 1)
        self.assertEqual(info.hits, 3)

        # change model in evalutor
        evaluator._model = build_model(game)

        # Still equal due to not clearing the cache
        value3 = evaluator.recurrent_evaluate(next_hidden)[0]
        self.assertEqual(value, value3)

        info = evaluator.recurrent_cache_info()
        self.assertEqual(info.misses, 1)
        self.assertEqual(info.hits, 4)

        evaluator.clear_cache()

        info = evaluator.recurrent_cache_info()
        self.assertEqual(info.misses, 0)
        self.assertEqual(info.hits, 0)

        # Now they differ from before
        value4 = evaluator.recurrent_evaluate(next_hidden)[0]
        value5 = evaluator.recurrent_evaluate(next_hidden)[0]
        self.assertNotEqual(value, value4)
        self.assertEqual(value4, value5)

        info = evaluator.recurrent_cache_info()
        self.assertEqual(info.misses, 1)
        self.assertEqual(info.hits, 1)

        self.assertEqual(np.array(next_hidden).tobytes(),
                         np.array(next_hidden2).tobytes())
        value6 = evaluator.recurrent_evaluate(next_hidden2)[0]
        self.assertEqual(value4, value6)

        info = evaluator.recurrent_cache_info()
        self.assertEqual(info.misses, 1)
        self.assertEqual(info.hits, 2)

    # def test_works_with_mcts(self):
    #     game = pyspiel.load_game("tic_tac_toe")
    #     model = build_model(game)
    #     evaluator = evaluator_lib.AlphaZeroEvaluator(game, model)
    #     bot = mcts.MCTSBot(
    #         game, 1., 20, evaluator, solve=False, dirichlet_noise=(0.25, 1.))
    #     root = bot.mcts_search(game.new_initial_state())
    #     self.assertEqual(root.explore_count, 20)


if __name__ == "__main__":
    absltest.main()
