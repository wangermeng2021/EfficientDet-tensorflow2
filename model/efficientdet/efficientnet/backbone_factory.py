# Copyright 2020 Google Research. All Rights Reserved.
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
# ==============================================================================
"""Backbone network factory."""
import os
from absl import logging
import tensorflow as tf

from backbone import efficientnet_builder
from backbone import efficientnet_lite_builder
from backbone import efficientnet_model



# min_level = 3
# num_levels =4
# node_ids = {min_level + i: [i] for i in range(num_levels)}
# print(node_ids)
# exit()

class Struct(object):
  """Comment removed"""

  def __init__(self, data):
    for name, value in data.items():
      setattr(self, name, self._wrap(value))

  def _wrap(self, value):
    if isinstance(value, (tuple, list, set, frozenset)):
      return type(value)([self._wrap(v) for v in value])
    else:
      return Struct(value) if isinstance(value, dict) else value




  # global_cfg = {
  #   'dropout_rate': 0.4,
  #   'num_classes': 1000,
  #   'width_coefficient': 1.6,
  #   'depth_coefficient': 2.2,
  #   'depth_divisor': 8,
  #   'min_depth': None,
  #   'survival_prob': 0.8,
  #   'batch_norm': 'bn',
  #   'blocks_args': ['r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
  #                   'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
  #                   'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
  #                   'r1_k3_s11_e6_i192_o320_se0.25']
  # }
BLOCK_CFG = [
  {
    'kernel_size': 3,
    'num_repeat': 1,
    'input_filters': 32,
    'output_filters': 16,
    'expand_ratio': 1,
    'strides': [1, 1],
    'se_ratio': 0.25,
  },
  {
    'kernel_size': 3,
    'num_repeat': 2,
    'input_filters': 16,
    'output_filters': 24,
    'expand_ratio': 6,
    'strides': [2, 2],
    'se_ratio': 0.25,
  },
  {
    'kernel_size': 5,
    'num_repeat': 2,
    'input_filters': 24,
    'output_filters': 40,
    'expand_ratio': 6,
    'strides': [2, 2],
    'se_ratio': 0.25,
  },
  {
    'kernel_size': 3,
    'num_repeat': 3,
    'input_filters': 40,
    'output_filters': 80,
    'expand_ratio': 6,
    'strides': [2, 2],
    'se_ratio': 0.25,
  },
  {
    'kernel_size': 5,
    'num_repeat': 3,
    'input_filters': 80,
    'output_filters': 112,
    'expand_ratio': 6,
    'strides': [1, 1],
    'se_ratio': 0.25,
  },
  {
    'kernel_size': 5,
    'num_repeat': 4,
    'input_filters': 112,
    'output_filters': 192,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': [2, 2],
    'se_ratio': 0.25,
  },
  {
    'kernel_size': 3,
    'num_repeat': 1,
    'input_filters': 192,
    'output_filters': 320,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': [1, 1],
    'se_ratio': 0.25,
  },
]
EFFICIENTNET_CFG = {
  'efficientnet-b0':
    {
      'width_coefficient': 1.0,
      'depth_coefficient': 1.0,
      'resolution': 224,
      'dropout_rate': 0.2,
      'depth_divisor': 8,
      'survival_prob': 0.,
      'num_classes': 1000,
    },
  'efficientnet-b1':
    {
      'width_coefficient': 1.0,
      'depth_coefficient': 1.1,
      'resolution': 240,
      'dropout_rate': 0.2,
      'depth_divisor': 8,
      'survival_prob': 0.8,
      'num_classes': 1000,
    },
  'efficientnet-b2':
    {
      'width_coefficient': 1.1,
      'depth_coefficient': 1.2,
      'resolution': 260,
      'dropout_rate': 0.3,
      'depth_divisor': 8,
      'survival_prob': 0.8,
      'num_classes': 1000,
    },
  'efficientnet-b3':
    {
      'width_coefficient': 1.2,
      'depth_coefficient': 1.4,
      'resolution': 300,
      'dropout_rate': 0.3,
      'depth_divisor': 8,
      'survival_prob': 0.8,
      'num_classes': 1000,
    },
  'efficientnet-b4':
    {
      'width_coefficient': 1.4,
      'depth_coefficient': 1.8,
      'resolution': 380,
      'dropout_rate': 0.4,
      'depth_divisor': 8,
      'survival_prob': 0.8,
      'num_classes': 1000,
    },
  'efficientnet-b5':
    {
      'width_coefficient': 1.6,
      'depth_coefficient': 2.2,
      'resolution': 456,
      'dropout_rate': 0.4,
      'depth_divisor': 8,
      'survival_prob': 0.8,
      'num_classes': 1000,
    },
  'efficientnet-b6':
    {
      'width_coefficient': 1.8,
      'depth_coefficient': 2.6,
      'resolution': 528,
      'dropout_rate': 0.5,
      'depth_divisor': 8,
      'survival_prob': 0.8,
      'num_classes': 1000,
    },
  'efficientnet-b7':
    {
      'width_coefficient': 2.0,
      'depth_coefficient': 3.1,
      'resolution': 600,
      'dropout_rate': 0.5,
      'depth_divisor': 8,
      'survival_prob': 0.8,
      'num_classes': 1000,
    },
  'efficientnet-b8':
    {
      'width_coefficient': 2.2,
      'depth_coefficient': 3.6,
      'resolution': 672,
      'dropout_rate': 0.5,
      'depth_divisor': 8,
      'survival_prob': 0.8,
      'num_classes': 1000,
    },
  'efficientnet-l2':
    {
      'width_coefficient': 4.3,
      'depth_coefficient': 5.3,
      'resolution': 800,
      'dropout_rate': 0.5,
      'depth_divisor': 8,
      'survival_prob': 0.8,
      'num_classes': 1000,
    },

}

import numpy as np
def get_model_args(model_name):
  cfgs = {'blocks':[]}
  for idx, block in enumerate(BLOCK_CFG):
    block['num_repeat'] = int(np.ceil(block['num_repeat']*EFFICIENTNET_CFG[model_name]['depth_coefficient']))
    block['output_filters'] = (block['output_filters'] * EFFICIENTNET_CFG[model_name]['width_coefficient']+EFFICIENTNET_CFG[model_name]['depth_divisor']/2)//EFFICIENTNET_CFG[model_name]['depth_divisor']*EFFICIENTNET_CFG[model_name]['depth_divisor']
    block['input_filters'] = (block['input_filters'] * EFFICIENTNET_CFG[model_name]['width_coefficient']+EFFICIENTNET_CFG[model_name]['depth_divisor']/2)//EFFICIENTNET_CFG[model_name]['depth_divisor']*EFFICIENTNET_CFG[model_name]['depth_divisor']
    # block['expand_ratio'] = block['expand_ratio']
    # block['se_ratio'] = block['se_ratio']
    # survival_prob = EFFICIENTNET_CFG[model_name]['survival_prob']
    # if survival_prob:
    #   drop_rate = 1.0 - survival_prob
    #   survival_prob = 1.0 - drop_rate * float(idx) / len(BLOCK_CFG)
    # block['survival_prob'] = block['survival_prob']
    cfgs['blocks'].append(block)
  cfgs['dropout_rate'] = EFFICIENTNET_CFG[model_name]['dropout_rate']
  cfgs['resolution'] = EFFICIENTNET_CFG[model_name]['resolution']
  cfgs['depth_divisor'] = EFFICIENTNET_CFG[model_name]['depth_divisor']
  cfgs['conv_head_filters'] = (1280 * EFFICIENTNET_CFG[model_name]['width_coefficient']+EFFICIENTNET_CFG[model_name]['depth_divisor']/2)//EFFICIENTNET_CFG[model_name]['depth_divisor']*EFFICIENTNET_CFG[model_name]['depth_divisor']
  cfgs['num_classes'] = EFFICIENTNET_CFG[model_name]['num_classes']
  cfgs['survival_prob'] = EFFICIENTNET_CFG[model_name]['survival_prob']
  return cfgs



def get_model_builder(model_name):
  """Get the model_builder module for a given model name."""
  if model_name.startswith('efficientnet-lite'):
    return efficientnet_lite_builder
  elif model_name.startswith('efficientnet-'):
    return efficientnet_builder
  else:
    raise ValueError('Unknown model name {}'.format(model_name))


def get_model(model_name, override_params=None, model_dir=None):
  """A helper function to create and return model.

  Args:
    model_name: string, the predefined model name.
    override_params: A dictionary of params for overriding. Fields must exist in
      efficientnet_model.GlobalParams.
    model_dir: string, optional model dir for saving configs.

  Returns:
    created model

  Raises:
    When model_name specified an undefined model, raises NotImplementedError.
    When override_params has invalid fields, raises ValueError.
  """

  # For backward compatibility.
  if override_params and override_params.get('drop_connect_rate', None):
    override_params['survival_prob'] = 1 - override_params['drop_connect_rate']

  if not override_params:
    override_params = {}

  if model_name.startswith('efficientnet-lite'):
    builder = efficientnet_lite_builder
  elif model_name.startswith('efficientnet-'):
    builder = efficientnet_builder
  else:
    raise ValueError('Unknown model name {}'.format(model_name))
  # print("iiiiiiiiiiiiiiiiiiiiiiiiiiiiii")
  # print(override_params)
  #blocks_args, global_params = builder.get_model_params(model_name,
  #                                                      override_params)
  # print(global_params)
  # if model_dir:
  #   param_file = os.path.join(model_dir, 'model_params.txt')
  #   if not tf.io.gfile.exists(param_file):
  #     if not tf.io.gfile.exists(model_dir):
  #       tf.io.gfile.mkdir(model_dir)
  #     with tf.io.gfile.GFile(param_file, 'w') as f:
  #       logging.info('writing to %s', param_file)
  #       f.write('model_name= %s\n\n' % model_name)
  #       f.write('global_params= %s\n\n' % str(global_params))
  #       f.write('blocks_args= %s\n\n' % str(blocks_args))

  # global_args = Struct(global_cfg)
  # blocks_args = [Struct(i) for i in block_cfg]
  model_name = 'efficientnet-b0'
  cfgs = get_model_args(model_name)
  return efficientnet_model.Model(cfgs,model_name)



