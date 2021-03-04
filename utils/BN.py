

import tensorflow as tf

class BatchNormalization(tf.keras.layers.BatchNormalization):

  def __init__(self, **kwargs):
    if not kwargs.get('name', None):
      kwargs['name'] = 'tpu_batch_normalization'
    super().__init__(**kwargs)

  def call(self, inputs, training=None):
    outputs = super().call(inputs, training)
    for u in self.updates:
      tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, u)
    return outputs


def get_bn(type):
  if type == 'bn':
    return BatchNormalization
  else:
    raise ValueError('unsupported bn type {}'.format(type))


