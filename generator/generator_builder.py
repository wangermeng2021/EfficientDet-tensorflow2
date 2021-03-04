
from generator.voc_generator import VocGenerator
from generator.coco_generator import CocoGenerator
def get_generator(args):
    if args.dataset_type == 'voc':
        train_dataset = VocGenerator(args, mode='train')
        val_dataset = VocGenerator(args, mode='val')
        pred_dataset  = VocGenerator(args, mode='pred')
    elif args.dataset_type == 'coco':
        train_dataset = CocoGenerator(args, mode='train')
        val_dataset = CocoGenerator(args, mode='val')
        pred_dataset  = CocoGenerator(args, mode='pred')
    else:
        raise ValueError("{} is invalid!".format(args.dataset_type))
    # return train_dataset, valid_dataset, pred_dataset
    return train_dataset, val_dataset, pred_dataset