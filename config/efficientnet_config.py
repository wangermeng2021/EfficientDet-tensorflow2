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
from utils.struct_config import Config
import numpy as np
def get_struct_args(model_name):
  cfgs = {'blocks':[]}
  for idx, block1 in enumerate(BLOCK_CFG):
    block = block1.copy()
    block['num_repeat'] = int(np.ceil(block['num_repeat']*EFFICIENTNET_CFG[model_name]['depth_coefficient']))
    block['output_filters'] = (block['output_filters'] * EFFICIENTNET_CFG[model_name]['width_coefficient']+EFFICIENTNET_CFG[model_name]['depth_divisor']/2)//EFFICIENTNET_CFG[model_name]['depth_divisor']*EFFICIENTNET_CFG[model_name]['depth_divisor']
    block['input_filters'] = (block['input_filters'] * EFFICIENTNET_CFG[model_name]['width_coefficient']+EFFICIENTNET_CFG[model_name]['depth_divisor']/2)//EFFICIENTNET_CFG[model_name]['depth_divisor']*EFFICIENTNET_CFG[model_name]['depth_divisor']
    cfgs['blocks'].append(block)

  cfgs['dropout_rate'] = EFFICIENTNET_CFG[model_name]['dropout_rate']
  cfgs['resolution'] = EFFICIENTNET_CFG[model_name]['resolution']
  cfgs['depth_divisor'] = EFFICIENTNET_CFG[model_name]['depth_divisor']
  cfgs['conv_head_filters'] = (1280 * EFFICIENTNET_CFG[model_name]['width_coefficient']+EFFICIENTNET_CFG[model_name]['depth_divisor']/2)//EFFICIENTNET_CFG[model_name]['depth_divisor']*EFFICIENTNET_CFG[model_name]['depth_divisor']
  cfgs['num_classes'] = EFFICIENTNET_CFG[model_name]['num_classes']
  cfgs['survival_prob'] = EFFICIENTNET_CFG[model_name]['survival_prob']
  return Config(cfgs)