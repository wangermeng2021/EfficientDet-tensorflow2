
# from labeler.scaled_yolov4_labeler import ScaledYolov4Labeler
from labeler.efficientdet_labeler import EfficientdetLabeler
def get_labels(args):
    if args.model_name == 'efficientdet':
        return EfficientdetLabeler(args)
    else:
        raise ValueError('not support {}'.format(args.model_name))