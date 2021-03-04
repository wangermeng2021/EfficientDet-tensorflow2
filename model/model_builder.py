
from model.efficientdet import efficientdet
def get_model(args,training=True):
    if args.model_name == "efficientdet":
        model = efficientdet.get_model(args, training=training)
    else:
        raise ValueError('unsupported model type {}'.format(args.model_name))
    return model