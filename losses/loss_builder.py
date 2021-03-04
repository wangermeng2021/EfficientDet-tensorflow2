
from losses import efficientdet_loss
def  get_loss(args):
    if args.model_name == "efficientdet":
        loss_fun = efficientdet_loss.get_loss(args)
    else:
        raise ValueError('unsupported model {}.'.format(args.model_name))
    return loss_fun
