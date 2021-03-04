
import tensorflow as tf



def get_optimizers(args):
    if args.optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=args.init_lr,momentum=args.momentum, nesterov=args.nesterov, name='sgd')
        args.moving_average_decay
    elif args.optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.init_lr,name='adam')
    else:
        raise ValueError("{} is not supported!".format(args.optimizer))
    return optimizer

