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
# =============================================================================
"""Postprocessing for anchor-based detection."""
import functools
from typing import List, Tuple

from absl import logging
import tensorflow as tf

import utils
from model.efficientdet import anchors
T = tf.Tensor  # a shortcut for typing check.
CLASS_OFFSET = 1

def to_list(inputs):
  if isinstance(inputs, dict):
    return [tf.cast(inputs[k], tf.float32) for k in sorted(inputs.keys())]
  if isinstance(inputs, list):
    return [tf.cast(i, tf.float32) for i in inputs]
  raise ValueError('Unrecognized inputs : {}'.format(inputs))


def batch_map_fn(map_fn, inputs, *args):
  """Apply map_fn at batch dimension."""
  if isinstance(inputs[0], (list, tuple)):
    batch_size = len(inputs[0])
  else:
    batch_size = inputs[0].shape.as_list()[0]

  if not batch_size:
    # handle dynamic batch size: tf.vectorized_map is faster than tf.map_fn.
    return tf.vectorized_map(map_fn, inputs, *args)

  outputs = []
  for i in range(batch_size):
    outputs.append(map_fn([x[i] for x in inputs]))
  return [tf.stack(y) for y in zip(*outputs)]


def clip_boxes(boxes: T, image_size: int) -> T:
  """Clip boxes to fit the image size."""
  image_size = utils.parse_image_size(image_size) * 2
  return tf.clip_by_value(boxes, [0], image_size)


def merge_class_box_level_outputs(args, cls_outputs: List[T],
                                  box_outputs: List[T]) -> Tuple[T, T]:
  """Concatenates class and box of all levels into one tensor."""
  cls_outputs_all, box_outputs_all = [], []
  batch_size = tf.shape(cls_outputs[0])[0]
  for level in range(0, args.max_level - args.min_level + 1):
    cls_outputs_all.append(
        tf.reshape(cls_outputs[level], [batch_size, -1, args.num_classes]))
    box_outputs_all.append(tf.reshape(box_outputs[level], [batch_size, -1, 4]))
  return tf.concat(cls_outputs_all, 1), tf.concat(box_outputs_all, 1)


def topk_class_boxes(params, cls_outputs: T,
                     box_outputs: T) -> Tuple[T, T, T, T]:
  """Pick the topk class and box outputs."""
  batch_size = tf.shape(cls_outputs)[0]

  # Keep all anchors, but for each anchor, just keep the max probablity for
  # each class.
  cls_outputs_idx = tf.math.argmax(cls_outputs, axis=-1, output_type=tf.int32)
  num_anchors = tf.shape(cls_outputs)[1]

  classes = cls_outputs_idx
  indices = tf.tile(
      tf.expand_dims(tf.range(num_anchors), axis=0), [batch_size, 1])
  cls_outputs_topk = tf.reduce_max(cls_outputs, -1)
  box_outputs_topk = box_outputs

  return cls_outputs_topk, box_outputs_topk, classes, indices

from config import efficientdet_config
def pre_nms(args, cls_outputs, box_outputs, topk=True):
  """Detection post processing before nms.

  It takes the multi-level class and box predictions from network, merge them
  into unified tensors, and compute boxes, scores, and classes.

  Args:
    params: a dict of parameters.
    cls_outputs: a list of tensors for classes, each tensor denotes a level of
      logits with shape [N, H, W, num_class * num_anchors].
    box_outputs: a list of tensors for boxes, each tensor ddenotes a level of
      boxes with shape [N, H, W, 4 * num_anchors].
    topk: if True, select topk before nms (mainly to speed up nms).

  Returns:
    A tuple of (boxes, scores, classes).
  """
  # get boxes by apply bounding box regression to anchors.
  model_args = efficientdet_config.get_struct_args(args)
  eval_anchors = anchors.Anchors(args.min_level, args.max_level,
                                 args.num_scales, args.aspect_ratios,
                                 args.anchor_scale, model_args.image_size)

  cls_outputs, box_outputs = merge_class_box_level_outputs(
      args, cls_outputs, box_outputs)

  if topk:
    # select topK purely based on scores before NMS, in order to speed up nms.
    cls_outputs, box_outputs, classes, indices = topk_class_boxes(
        args, cls_outputs, box_outputs)
    anchor_boxes = tf.gather(eval_anchors.boxes, indices)
  else:
    anchor_boxes = eval_anchors.boxes
    classes = None

  boxes = anchors.decode_box_outputs(box_outputs, anchor_boxes)
  # convert logits to scores.
  scores = tf.math.sigmoid(cls_outputs)
  return boxes, scores, classes

def postprocess(args, cls_outputs, box_outputs,image_size):
  """Post processing with combined NMS.

  Leverage the tf combined NMS. It is fast on TensorRT, but slow on CPU/GPU.

  Args:
    params: a dict of parameters.
    cls_outputs: a list of tensors for classes, each tensor denotes a level of
      logits with shape [N, H, W, num_class * num_anchors].
    box_outputs: a list of tensors for boxes, each tensor ddenotes a level of
      boxes with shape [N, H, W, 4 * num_anchors]. Each box format is [y_min,
      x_min, y_max, x_man].
    image_scales: scaling factor or the final image and bounding boxes.

  Returns:
    A tuple of batch level (boxes, scores, classess, valid_len) after nms.
  """
  cls_outputs = to_list(cls_outputs)
  box_outputs = to_list(box_outputs)
  # Don't filter any outputs because combine_nms need the raw information.
  boxes, scores, _ = pre_nms(args, cls_outputs, box_outputs, topk=False)
  nms_boxes, nms_scores, nms_cls, nms_valid_len = (
      tf.image.combined_non_max_suppression(
          tf.expand_dims(boxes, axis=2),
          scores,
          args.nms_max_box_num,
          args.nms_max_box_num,
          score_threshold=args.nms_score_threshold,
          clip_boxes=False))
  CLASS_OFFSET = 0
  nms_cls += CLASS_OFFSET

  nms_boxes =  tf.clip_by_value(nms_boxes,[0], tf.tile(image_size,[2]))
  # nms_boxes = tf.identity(nms_boxes, name="output_boxes")
  # nms_scores = tf.identity(nms_scores, name="output_scores")
  # nms_cls = tf.identity(nms_cls, name="output_cls")
  return nms_boxes, nms_scores, nms_cls, nms_valid_len

