
from losses.losses import BoxRegressionLoss,ClassificationLoss
import tensorflow as tf
from model.efficientdet import anchors
#
# https://github.com/google/automl/tree/master/efficientdet
# the implementation of offical automl's efficientdet



def get_cls_loss(args,index):
    def loss(y_true,y_pred):
        if args.cls_loss == 'focal':
            cls_loss_fun = ClassificationLoss.focal(args.focal_alpha, args.focal_gamma,args.label_smooth)
        else:
            raise ValueError('unsupported loss type {}'.format(args.cls_loss))
        y_pred_cls = tf.reshape(y_pred, [-1])
        y_true_cls = tf.reshape(y_true,[-1])
        y_true_mask = tf.cast(tf.not_equal(y_true_cls, -1), tf.dtypes.float32)
        num_positive_anchor = tf.cast(tf.maximum(tf.reduce_sum(y_true_mask),1),tf.dtypes.float32)

        y_true_cls_one_hot = tf.one_hot(y_true_cls,args.num_classes,dtype=y_true_cls.dtype)
        y_true_cls_one_hot = tf.cast(tf.reshape(y_true_cls_one_hot, [-1]),tf.dtypes.float32)

        cls_loss = cls_loss_fun(y_true_cls_one_hot,y_pred_cls)/num_positive_anchor

        return cls_loss
    return loss

def get_box_loss(args,index):
    def loss(y_true,y_pred):
        # box_iou_loss_fun = BoxRegressionLoss.efficientdet_iou_loss(type=train_args.loss.box_iou_loss)
        if args.box_loss == 'huber':
            box_loss_fun = BoxRegressionLoss.huber()
        else:
            raise ValueError('unsupported loss type {}'.format(args.box_loss))

        ####
        # input_anchors = anchors.Anchors(model_args.min_level, model_args.max_level, model_args.num_scales,
        #                                 model_args.aspect_ratios, model_args.anchor_scale,
        #                                 model_args.image_size)

        y_pred_boxes = tf.reshape(y_pred, [-1, 4])
        y_true_boxes = tf.reshape(y_true, [-1, 4])
        y_true_mask = tf.cast(tf.not_equal(y_true_boxes[...,0],-1), tf.dtypes.float32)
        num_positive_anchor = tf.maximum(tf.reduce_sum(y_true_mask), 1)

        # anchor_boxes = tf.tile(input_anchors.boxes_levels[index], [tf.shape(y_pred_boxes)[0] // input_anchors.boxes_levels[index].shape[0], 1])
        # y_pred_x1y1x2y2 = anchors.decode_box_outputs(y_pred_boxes, anchor_boxes)
        # y_true_x1y1x2y2 = anchors.decode_box_outputs(y_true_boxes, anchor_boxes)
        # y_pred_x1y1x2y2 = tf.boolean_mask(y_pred_x1y1x2y2, y_true_mask)
        # y_true_x1y1x2y2 = tf.boolean_mask(y_true_x1y1x2y2, y_true_mask)

        y_pred_x1y1x2y2 = tf.boolean_mask(y_pred_boxes, y_true_mask)
        y_true_x1y1x2y2 = tf.boolean_mask(y_true_boxes, y_true_mask)
        box_loss = box_loss_fun(y_true_x1y1x2y2, y_pred_x1y1x2y2)/(num_positive_anchor*4)

        # box_iou_loss = box_iou_loss_fun(y_true_x1y1x2y2,y_pred_x1y1x2y2)/(num_positive_anchor)

        return box_loss
    return loss

def get_loss(args):
    num_level = args.max_level-args.min_level+1
    cls_loss_funs = []
    box_loss_funs = []
    for i in range(num_level):
        cls_loss_funs.append(get_cls_loss(args, i))
        box_loss_funs.append(get_box_loss(args, i))

    return (cls_loss_funs,box_loss_funs)



