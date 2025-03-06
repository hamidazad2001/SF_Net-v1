# Copyright 2022 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A library for instantiating the model for training frame interpolation.

All models are expected to use three inputs: input image batches 'x0' and 'x1'
and 'time', the fractional time where the output should be generated.

The models are expected to output the prediction as a dictionary that contains
at least the predicted image batch as 'image' plus optional data for debug,
analysis or custom losses.
"""

import gin.tf
from src.models import interpolator 
from src.utils import options as my_options

import tensorflow as tf


@gin.configurable('model')
def create_model(name: str) -> tf.keras.Model:
  """Creates the frame interpolation model based on given model name."""
  print(name)
  if name == 'film_net':
    return _create_model()  # pylint: disable=no-value-for-parameter
  elif name == 'azadegan_net':
    return _create_azadegan_net_model()  # pylint: disable=no-value-for-parameter
  elif name == 'both_net':
    return _create_both_net_model()

  else:
    raise ValueError(f'Model {name} not implemented.')


def _create_model(net_flag='filmNet') -> tf.keras.Model:
  """Creates the film_net interpolator."""
  # Options are gin-configured in the Options class directly.
  options = my_options.Options()

  x0 = tf.keras.Input(
      shape=(256, 256, 3), batch_size=None, dtype=tf.float32, name='x0')
  x1 = tf.keras.Input(
      shape=(256, 256, 3), batch_size=None, dtype=tf.float32, name='x1')
  time = tf.keras.Input(
      shape=(1,), batch_size=None, dtype=tf.float32, name='time')

  return interpolator.create_model(x0, x1, time, options, net_flag)


def _create_both_net_model() -> tf.keras.Model:
  """Creates the film_net interpolator."""
  # Options are gin-configured in the Options class directly.
  return _create_model(net_flag='both')
  


def _create_azadegan_net_model() -> tf.keras.Model:
  return _create_model(net_flag='azadegan')
