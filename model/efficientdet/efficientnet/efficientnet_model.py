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
"""Contains definitions for EfficientNet model.

[1] Mingxing Tan, Quoc V. Le
  EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
  ICML'19, https://arxiv.org/abs/1905.11946
"""

import collections
import itertools
import math

from absl import logging
import numpy as np
import six
from six.moves import xrange
import tensorflow as tf
from model.efficientdet import utils
import copy
GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate', 'data_format',
    'num_classes', 'width_coefficient', 'depth_coefficient', 'depth_divisor',
    'min_depth', 'survival_prob', 'relu_fn', 'batch_norm', 'use_se',
    'local_pooling', 'condconv_num_experts', 'clip_projection_output',
    'blocks_args', 'fix_head_stem',
])
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)

BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'strides', 'se_ratio', 'conv_type', 'fused_conv',
    'super_pixel', 'condconv'
])
# defaults will be a public argument for namedtuple in Python 3.7
# https://docs.python.org/3/library/collections.html#collections.namedtuple
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


def conv_kernel_initializer(shape, dtype=None, partition_info=None):
  """Initialization for convolutional kernels.

  The main difference with tf.variance_scaling_initializer is that
  tf.variance_scaling_initializer uses a truncated normal with an uncorrected
  standard deviation, whereas here we use a normal distribution. Similarly,
  tf.initializers.variance_scaling uses a truncated normal with
  a corrected standard deviation.

  Args:
    shape: shape of variable
    dtype: dtype of variable
    partition_info: unused

  Returns:
    an initialization for the variable
  """
  del partition_info
  kernel_height, kernel_width, _, out_filters = shape
  fan_out = int(kernel_height * kernel_width * out_filters)
  return tf.random.normal(
      shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype)


def dense_kernel_initializer(shape, dtype=None, partition_info=None):
  """Initialization for dense kernels.

  This initialization is equal to
    tf.variance_scaling_initializer(scale=1.0/3.0, mode='fan_out',
                                    distribution='uniform').
  It is written out explicitly here for clarity.

  Args:
    shape: shape of variable
    dtype: dtype of variable
    partition_info: unused

  Returns:
    an initialization for the variable
  """
  del partition_info
  init_range = 1.0 / np.sqrt(shape[1])
  return tf.random.uniform(shape, -init_range, init_range, dtype=dtype)


# def round_filters(filters, global_params, skip=False):
#   """Round number of filters based on depth multiplier."""
#   multiplier = global_params.width_coefficient
#   divisor = global_params.depth_divisor
#   min_depth = global_params.min_depth
#   if skip or not multiplier:
#     print("kkkkkkkkkkkkkkkkkkkkkkkk")
#     return filters
#
#   filters *= multiplier
#   min_depth = min_depth or divisor
#   new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
#   # Make sure that round down does not go down by more than 10%.
#   if new_filters < 0.9 * filters:
#     new_filters += divisor
#
#   return int(new_filters)


# def round_repeats(repeats, global_params, skip=False):
#   """Round number of filters based on depth multiplier."""
#   multiplier = global_params.depth_coefficient
#   if skip or not multiplier:
#
#     return repeats
#   return int(math.ceil(multiplier * repeats))


class SE(tf.keras.layers.Layer):
  """Squeeze-and-excitation layer."""

  def __init__(self, se_filters, output_filters, name=None):
    super().__init__(name=name)

    self._relu_fn = tf.nn.swish

    # Squeeze and Excitation layer.
    self._se_reduce = tf.keras.layers.Conv2D(
        se_filters,
        kernel_size=[1, 1],
        strides=[1, 1],
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        use_bias=True,
        name='conv2d')
    self._se_expand = tf.keras.layers.Conv2D(
        output_filters,
        kernel_size=[1, 1],
        strides=[1, 1],
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        use_bias=True,
        name='conv2d_1')

  def call(self, inputs):

    se_tensor = tf.reduce_mean(inputs, [1, 2], keepdims=True)
    se_tensor = self._se_expand(self._relu_fn(self._se_reduce(se_tensor)))
    return tf.sigmoid(se_tensor) * inputs

class MBConvBlock(tf.keras.layers.Layer):
  """A class of MBConv: Mobile Inverted Residual Bottleneck.

  Attributes:
    endpoints: dict. A list of internal tensors.
  """

  def __init__(self, block_args, name=None):
    """Initializes a MBConv block.
    Args:
      block_args: BlockArgs, arguments to create a Block.
      global_params: GlobalParams, a set of global parameters.
      name: layer name.
    """
    super().__init__(name=name)

    self._block_args = block_args
    # self._batch_norm =tf.keras.layers.BatchNormalization
    self._batch_norm =utils.BatchNormalization
    self._relu_fn = tf.nn.swish
    self.endpoints = None

    # Builds the block accordings to arguments.
    self._build()
  def _build(self):
    """Builds block according to the arguments."""
    bid = itertools.count(0)
    get_bn_name = lambda: 'tpu_batch_normalization' + ('' if not next(
        bid) else '_' + str(next(bid) // 2))
    cid = itertools.count(0)
    get_conv_name = lambda: 'conv2d' + ('' if not next(cid) else '_' + str(
        next(cid) // 2))

      # Expansion phase. Called if not using fused convolutions and expansion
      # phase is necessary.
    expand_filters = self._block_args['input_filters']*self._block_args['expand_ratio']
    se_filters = self._block_args['input_filters']*self._block_args['se_ratio']
    if self._block_args['expand_ratio'] != 1:
      self._expand_conv = tf.keras.layers.Conv2D(
            filters=expand_filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            use_bias=False,
            name=get_conv_name())
      self._bn0 = self._batch_norm(name=get_bn_name())
      # Depth-wise convolution phase. Called if not using fused convolutions.
    self._depthwise_conv = tf.keras.layers.DepthwiseConv2D(
          kernel_size=[self._block_args['kernel_size'], self._block_args['kernel_size']],
          strides=self._block_args['strides'],
          depthwise_initializer=conv_kernel_initializer,
          padding='same',
          use_bias=False,
          name='depthwise_conv2d')

    self._bn1 = self._batch_norm(name=get_bn_name())
    self._se = SE(se_filters, expand_filters, name='se')
    # Output phase.
    self._project_conv = tf.keras.layers.Conv2D(
        filters=self._block_args['output_filters'],
        kernel_size=[1, 1],
        strides=[1, 1],
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        use_bias=False,
        name=get_conv_name())
    self._bn2 = self._batch_norm(name=get_bn_name())

  def call(self, inputs, training, survival_prob=None):
    """Implementation of call().

    Args:
      inputs: the inputs tensor.
      training: boolean, whether the model is constructed for training.
      survival_prob: float, between 0 to 1, drop connect rate.

    Returns:
      A output tensor.
    """
    def _call(inputs):
      x = inputs
      # Otherwise, first apply expansion and then apply depthwise conv.
      if self._block_args['expand_ratio'] != 1:
          x = self._relu_fn(self._bn0(self._expand_conv(x), training=training))
      x = self._relu_fn(self._bn1(self._depthwise_conv(x), training=training))
      x = self._se(x)
      self.endpoints = {'expansion_output': x}
      x = self._bn2(self._project_conv(x), training=training)
      # Add identity so that quantization-aware training can insert quantization
      # ops correctly.
      x = tf.identity(x)
      if all(s == 1 for s in self._block_args['strides']) and self._block_args['input_filters'] == self._block_args['output_filters']:
          # Apply only if skip connection presents.
        if survival_prob:
          x = utils.drop_connect(x, training, survival_prob)
        x = tf.add(x, inputs)
      return x

    return _call(inputs)


class conv2d_bn_act(tf.keras.layers.Layer):
  def __init__(self,filters,bn='bn',act='swish',name=None):
    super().__init__(name=name)
    self._conv = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=[3, 3],
        strides=[2, 2],
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        use_bias=False,name='conv2d')
    if bn == 'bn':
        self._bn = utils.BatchNormalization()
    else:
        raise ValueError('{} is not supported!'.format(act))
    if act == 'swish':
        self._act = tf.nn.swish
    else:
        raise ValueError('{} is not supported!'.format(act))

  def call(self, inputs, training):
    return self._act(self._bn(self._conv(inputs), training=training))

class Head(tf.keras.layers.Layer):
  """Head layer for network outputs."""

  def __init__(self, cfgs, name=None):
    super().__init__(name=name)

    self.endpoints = {}
    self._cfgs = cfgs

    self._conv_head = tf.keras.layers.Conv2D(
        filters=self._cfgs['conv_head_filters'],
        kernel_size=[1, 1],
        strides=[1, 1],
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        use_bias=False,
        name='conv2d')
    # self._bn = tf.keras.layers.BatchNormalization
    self._bn = utils.BatchNormalization()
    self._relu_fn = tf.nn.swish

    self._avg_pooling = tf.keras.layers.GlobalAveragePooling2D()
    if self._cfgs['num_classes']:
      self._fc = tf.keras.layers.Dense(
          self._cfgs['num_classes'],
          kernel_initializer=dense_kernel_initializer)
    else:
      self._fc = None

    if self._cfgs['dropout_rate'] > 0:
      self._dropout = tf.keras.layers.Dropout(self._cfgs['dropout_rate'])
    else:
      self._dropout = None


  def call(self, inputs, training, pooled_features_only):
    """Call the layer."""
    outputs = self._relu_fn(
        self._bn(self._conv_head(inputs), training=training))
    self.endpoints['head_1x1'] = outputs

    outputs = self._avg_pooling(outputs)
    self.endpoints['pooled_features'] = outputs
    if not pooled_features_only:
        if self._dropout:
          outputs = self._dropout(outputs, training=training)
        self.endpoints['global_pool'] = outputs
        if self._fc:
          outputs = self._fc(outputs)
        self.endpoints['head'] = outputs
    return outputs


class Model(tf.keras.Model):

  def __init__(self, cfgs, name=None):
    """Initializes an `Model` instance.

    Args:
      blocks_args: A list of BlockArgs to construct block modules.
      global_params: GlobalParams, a set of global parameters.
      name: A string of layer name.

    Raises:
      ValueError: when blocks_args is not specified as a list.
    """
    super().__init__(name=name)

    self._cfgs = cfgs
    self._relu_fn =  tf.nn.swish
    # self._batch_norm = utils.BatchNormalization
    self.endpoints = None
    self._build()

  def _build(self):
    """Builds a model."""
    self._blocks = []
    self._stem = conv2d_bn_act(self._cfgs['blocks'][0]['input_filters'],bn='bn',act='swish',name='stem')

    block_id = itertools.count(0)
    block_name = lambda: 'blocks_%d' % next(block_id)
    for i, block_args in enumerate(self._cfgs['blocks']):
      block_args_copy = copy.deepcopy(block_args)
      self._blocks.append(MBConvBlock(block_args_copy, name=block_name()))
      if block_args['num_repeat'] > 1:
          for _ in xrange(block_args['num_repeat'] - 1):
            block_args_copy = copy.deepcopy(block_args)
            block_args_copy['input_filters']=block_args_copy['output_filters']
            block_args_copy['strides'] = [1, 1]
            self._blocks.append(MBConvBlock(block_args_copy, name=block_name()))

    # Head part.
    self._head = Head(self._cfgs)

  def call(self,
           inputs,
           training,
           features_only=None,
           pooled_features_only=False):
    """Implementation of call().

    Args:
      inputs: input tensors.
      training: boolean, whether the model is constructed for training.
      features_only: build the base feature network only.
      pooled_features_only: build the base network for features extraction
        (after 1x1 conv layer and global pooling, but before dropout and fc
        head).

    Returns:
      output tensors.
    """
    outputs = None
    self.endpoints = {}
    reduction_idx = 0

    # Calls Stem layers
    outputs = self._stem(inputs, training)

    self.endpoints['stem'] = outputs

    # Calls blocks.
    for idx, block in enumerate(self._blocks):
      is_reduction = False  # reduction flag for blocks after the stem layer
      # If the first block has super-pixel (space-to-depth) layer, then stem is
      # the first reduction point.

      if ((idx == len(self._blocks) - 1) or self._blocks[idx + 1]._block_args['strides'][0] > 1):
        is_reduction = True
        reduction_idx += 1
      survival_prob = self._cfgs['survival_prob']
      if survival_prob:
        drop_rate = 1.0 - survival_prob
        survival_prob = 1.0 - drop_rate * float(idx) / len(self._blocks)
      outputs = block(outputs, training=training, survival_prob=survival_prob)
      self.endpoints['block_%s' % idx] = outputs
      if is_reduction:
        self.endpoints['reduction_%s' % reduction_idx] = outputs
      if block.endpoints:
        for k, v in six.iteritems(block.endpoints):
          self.endpoints['block_%s/%s' % (idx, k)] = v
          if is_reduction:
            self.endpoints['reduction_%s/%s' % (reduction_idx, k)] = v
    self.endpoints['features'] = outputs
    if not features_only:
      # Calls final layers and returns logits.
      outputs = self._head(outputs, training, pooled_features_only)
      self.endpoints.update(self._head.endpoints)
    return [outputs] + list(
        filter(lambda endpoint: endpoint is not None, [
            self.endpoints.get('reduction_1'),
            self.endpoints.get('reduction_2'),
            self.endpoints.get('reduction_3'),
            self.endpoints.get('reduction_4'),
            self.endpoints.get('reduction_5'),
        ]))
