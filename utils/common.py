

from config.scaled_yolov4_config import CFG as scaled_yolov4_cfg

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

def cfg_to_struct(train_cfgs):
    args_list = []
    for train_cfg in train_cfgs:
        train_args = Struct(train_cfg)
        if train_args.model_name=='scaled_yolov4':
            scaled_yolov4_args = Struct(scaled_yolov4_cfg)
            args_list.append((train_args,scaled_yolov4_args))
    return args_list

