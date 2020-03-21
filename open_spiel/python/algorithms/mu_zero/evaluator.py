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
"""An MCTS Evaluator for an MuZero model."""

import numpy as np
import tensorflow.compat.v1 as tf

import pyspiel
from open_spiel.python.algorithms.mu_zero import muzero_mcts
from open_spiel.python.algorithms.mu_zero import model as model_lib
from open_spiel.python.utils import lru_cache


class MuZeroEvaluator(muzero_mcts.Evaluator):
    """An AlphaZero MCTS Evaluator."""

    def __init__(self, game, model: model_lib.Model, cache_size=2**16):
        """An AlphaZero MCTS Evaluator."""
        game_type = game.get_type()
        if game_type.reward_model != pyspiel.GameType.RewardModel.TERMINAL:
            raise ValueError("Game must have terminal rewards.")
        if game_type.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
            raise ValueError("Game must have sequential turns.")
        if game_type.chance_mode != pyspiel.GameType.ChanceMode.DETERMINISTIC:
            raise ValueError("Game must be deterministic.")

        self.num_actions = game.num_distinct_actions()
        self._model = model
        # use two cache to avoid same key in two inference function
        self._initial_cache = lru_cache.LRUCache(cache_size)
        self._recurrent_cache = lru_cache.LRUCache(cache_size)

    def initial_cache_info(self):
        return self._initial_cache.info()

    def recurrent_cache_info(self):
        return self._recurrent_cache.info()

    def clear_cache(self):
        self._initial_cache.clear()
        self._recurrent_cache.clear()

    def _initial_inference(self, state):
        obs = np.expand_dims(state.observation_tensor(), 0)

        # ndarray isn't hashable
        cache_key = obs.tobytes()

        value, reward, policy_logits, next_hidden = self._initial_cache.make(
            cache_key, lambda: self._model.initial_inference(obs))

        return value[0, 0], policy_logits[0]  # Unpack batch

    def _recurrent_inference(self, next_hidden):
        # tf.tensor is hashable
        # but the hash value of two tensor with same data is not the same
        cache_key = np.array(next_hidden).tobytes()

        value, reward, policy_logits, next_hidden = self._recurrent_cache.make(
            cache_key, lambda: self._model.recurrent_inference(next_hidden)
        )

        return value[0, 0], policy_logits[0]  # Unpack batch

    def initial_evaluate(self, state):
        """Returns a value for the given state."""
        value, _, = self._initial_inference(state)
        return np.array([value, -value])

    def initial_prior(self, state):
        """Returns the probabilities for all actions."""
        _, policy = self._initial_inference(state)
        return [(action, policy[action]) for action in state.legal_actions()]

    def recurrent_evaluate(self, next_hidden):
        """Returns a value for the given hidden state."""
        value, _, = self._recurrent_inference(next_hidden)
        return np.array([value, -value])

    def recurrent_prior(self, next_hidden):
        """Returns the probabilities for all actions."""
        _, policy = self._recurrent_inference(next_hidden)
        # need to find action space for game
        return [(action, policy[action]) for action in range(self.num_actions)]
