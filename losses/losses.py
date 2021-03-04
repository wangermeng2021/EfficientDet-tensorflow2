
import tensorflow as tf

import sys
import math
from model.efficientdet import anchors
class BoxRegressionLoss():

    @staticmethod
    def ciou(type,scales_x_y1,anchors,stride):
        def loss(y_true, y_pred):
            obj_mask = y_true[..., 4]
            pos_num = tf.reduce_sum(obj_mask,axis=[1,2,3])
            pos_num = tf.maximum(pos_num, 1)

            grid_height = tf.shape(y_true)[1]
            grid_width = tf.shape(y_true)[2]

            grid_xy = tf.stack(tf.meshgrid(tf.range(grid_width), tf.range(grid_height)), axis=-1)
            grid_xy = tf.reshape(grid_xy, [tf.shape(grid_xy)[0], tf.shape(grid_xy)[1], 1, tf.shape(grid_xy)[2]])
            grid_xy = tf.cast(grid_xy, tf.dtypes.float32)

            y_true_cxcy = grid_xy+y_true[...,0:2]
            y_true_x1y1x2y2 = tf.concat([y_true_cxcy - y_true[..., 2:4] / 2,y_true_cxcy + y_true[..., 2:4] / 2],axis=-1)

            scales_x_y = tf.cast(scales_x_y1,tf.dtypes.float32)
            scaled_pred_xy = tf.sigmoid(y_pred[..., 0:2]) * scales_x_y - 0.5 * (scales_x_y - 1)
            # pred_xy = (grid_xy + scaled_pred_xy)/(grid_width, grid_height)
            pred_xy = (grid_xy + scaled_pred_xy)

            # normalized_anchors = tf.cast(yolo_anchors[args.model_type][grid_index], tf.dtypes.float32) / tf.cast(
            #     8 * (2 ** grid_index) *tf.shape(y_true)[1:3][::-1], tf.dtypes.float32)
            # pred_wh = (tf.sigmoid(y_pred[..., 2:4]) * 2) ** 2 * normalized_anchors
            normalized_anchors = tf.cast(anchors, tf.dtypes.float32) / tf.cast(stride, tf.dtypes.float32)

            pred_wh = (tf.sigmoid(y_pred[..., 2:4]) * 2) ** 2 * normalized_anchors

            # return tf.reduce_sum(tf.abs(scaled_pred_xy - y_true[..., 0:2]) + tf.abs(pred_wh - y_true[..., 2:4]),[1,2,3,4])/pos_num,1

            pred_x1y1x2y2 = tf.concat([pred_xy-pred_wh/2, pred_xy+pred_wh/2], axis=-1)


            #iou
            inter_w = tf.maximum(tf.minimum(pred_x1y1x2y2[..., 2], y_true_x1y1x2y2[..., 2]) - tf.maximum(pred_x1y1x2y2[..., 0], y_true_x1y1x2y2[..., 0]), 0)
            inter_h = tf.maximum(tf.minimum(pred_x1y1x2y2[..., 3], y_true_x1y1x2y2[..., 3]) - tf.maximum(pred_x1y1x2y2[..., 1], y_true_x1y1x2y2[..., 1]), 0)
            inter_area = inter_w * inter_h
            pred_wh = pred_x1y1x2y2[..., 2:4] - pred_x1y1x2y2[..., 0:2]
            y_true_wh = y_true_x1y1x2y2[..., 2:4] - y_true_x1y1x2y2[..., 0:2]
            boxes1_area = pred_wh[...,0] * pred_wh[...,1]
            boxes2_area = y_true_wh[..., 0] * y_true_wh[..., 1]
            union_area = boxes1_area + boxes2_area - inter_area + 1e-07

            iou_scores = inter_area / union_area

            box_center_dist = (pred_x1y1x2y2[..., 0:2] + pred_x1y1x2y2[..., 2:4])/2-(y_true_x1y1x2y2[..., 0:2] + y_true_x1y1x2y2[..., 2:4]) / 2
            box_center_dist = tf.reduce_sum(tf.math.square(box_center_dist), axis=-1)
            bounding_rect_wh = tf.maximum(pred_x1y1x2y2[..., 2:4], y_true_x1y1x2y2[..., 2:4])-tf.minimum(pred_x1y1x2y2[..., 0:2], y_true_x1y1x2y2[..., 0:2])
            diagonal_line_length = tf.reduce_sum(tf.math.square(bounding_rect_wh),axis=-1)
            diou_loss = iou_scores - box_center_dist/(diagonal_line_length+1e-07)

            # return tf.reduce_sum(obj_mask * (1 - iou_scores), [1, 2, 3]) / pos_num,iou_scores

            # type = 'giou'
            if type == 'giou':
                bounding_rect_area = bounding_rect_wh[..., 0] * bounding_rect_wh[...,1] + 1e-07
                giou_loss = iou_scores - (bounding_rect_area-union_area)/bounding_rect_area
                return tf.reduce_sum(obj_mask*(1-giou_loss),[1,2,3])/pos_num, giou_loss
            if type == 'diou':
                return tf.reduce_sum(obj_mask*(1-diou_loss),[1,2,3])/pos_num, diou_loss

            y_true_wh = y_true_x1y1x2y2[..., 2:4] - y_true_x1y1x2y2[..., 0:2]
            v = (4./math.pi**2)*tf.square(tf.math.atan2(pred_wh[..., 1], pred_wh[...,0])-tf.math.atan2(y_true_wh[..., 1], y_true_wh[...,0]))

            alpha = v/(1.-iou_scores+v+1e-07)

            # alpha = v * tf.stop_gradient(pred_wh[..., 0] * pred_wh[..., 0], pred_wh[..., 1] * pred_wh[..., 1])
            alpha = tf.stop_gradient(alpha)
            ciou_loss = diou_loss - alpha*v
            return tf.reduce_sum(obj_mask*(1 - ciou_loss), [1,2,3])/pos_num,ciou_loss
        return loss

    def huber(delta=0.1):
        def loss(y_true, y_pred):
            y_true = tf.expand_dims(y_true, axis=-1)
            y_pred = tf.expand_dims(y_pred, axis=-1)
            huber_loss = tf.keras.losses.Huber(delta, reduction=tf.keras.losses.Reduction.NONE)(y_true, y_pred)
            # mask = tf.squeeze(tf.cast(tf.not_equal(y_true,0.0),tf.dtypes.float32),axis=-1)
            return tf.reduce_sum(huber_loss)
        return loss

    def efficientdet_iou_loss(type):
        def loss(y_true_x1y1x2y2, pred_x1y1x2y2):

            # pred_x1y1x2y2=tf.reshape(pred_x1y1x2y2,[tf.shape(pred_x1y1x2y2)[0],tf.shape(pred_x1y1x2y2)[1],tf.shape(pred_x1y1x2y2)[2], -1, 4])
            # y_true_x1y1x2y2 = tf.reshape(y_true_x1y1x2y2, [tf.shape(y_true_x1y1x2y2)[0], tf.shape(y_true_x1y1x2y2)[1],tf.shape(y_true_x1y1x2y2)[2], -1, 4])

            # obj_mask = y_true_x1y1x2y2[...,2:4]>y_true_x1y1x2y2[...,0:2]
            # obj_mask = tf.logical_and(obj_mask[...,0],obj_mask[...,1])
            # obj_mask = tf.cast(obj_mask,tf.dtypes.float32)
            # obj_mask_expand = tf.expand_dims(obj_mask,axis=-1)
            # pred_x1y1x2y2 = pred_x1y1x2y2 * obj_mask_expand
            # y_true_x1y1x2y2 = y_true_x1y1x2y2 * obj_mask_expand
            #iou
            inter_w = tf.maximum(tf.minimum(pred_x1y1x2y2[..., 2], y_true_x1y1x2y2[..., 2]) - tf.maximum(pred_x1y1x2y2[..., 0], y_true_x1y1x2y2[..., 0]), 0)
            inter_h = tf.maximum(tf.minimum(pred_x1y1x2y2[..., 3], y_true_x1y1x2y2[..., 3]) - tf.maximum(pred_x1y1x2y2[..., 1], y_true_x1y1x2y2[..., 1]), 0)
            inter_area = inter_w * inter_h
            pred_wh = pred_x1y1x2y2[..., 2:4] - pred_x1y1x2y2[..., 0:2]
            y_true_wh = y_true_x1y1x2y2[..., 2:4] - y_true_x1y1x2y2[..., 0:2]
            boxes1_area = pred_wh[...,0] * pred_wh[...,1]
            boxes2_area = y_true_wh[..., 0] * y_true_wh[..., 1]
            union_area = boxes1_area + boxes2_area - inter_area + 1e-07

            iou_scores = inter_area / union_area
            if type == 'iou':
                return tf.reduce_sum((1 - iou_scores))
            box_center_dist = (pred_x1y1x2y2[..., 0:2] + pred_x1y1x2y2[..., 2:4])/2-(y_true_x1y1x2y2[..., 0:2] + y_true_x1y1x2y2[..., 2:4]) / 2
            box_center_dist = tf.reduce_sum(tf.math.square(box_center_dist), axis=-1)
            bounding_rect_wh = tf.maximum(pred_x1y1x2y2[..., 2:4], y_true_x1y1x2y2[..., 2:4])-tf.minimum(pred_x1y1x2y2[..., 0:2], y_true_x1y1x2y2[..., 0:2])
            diagonal_line_length = tf.reduce_sum(tf.math.square(bounding_rect_wh),axis=-1)
            diou_loss = iou_scores - box_center_dist/(diagonal_line_length+1e-07)

            if type == 'giou':
                bounding_rect_area = bounding_rect_wh[..., 0] * bounding_rect_wh[...,1] + 1e-07
                giou_loss = iou_scores - (bounding_rect_area-union_area)/bounding_rect_area
                return tf.reduce_sum((1-giou_loss))
            if type == 'diou':
                return tf.reduce_sum((1-diou_loss))

            y_true_wh = y_true_x1y1x2y2[..., 2:4] - y_true_x1y1x2y2[..., 0:2]
            v = (4./math.pi**2)*tf.square(tf.math.atan2(pred_wh[..., 1], pred_wh[...,0])-tf.math.atan2(y_true_wh[..., 1], y_true_wh[...,0]))

            alpha = v/(1.-iou_scores+v+1e-07)

            # alpha = v * tf.stop_gradient(pred_wh[..., 0] * pred_wh[..., 0], pred_wh[..., 1] * pred_wh[..., 1])
            alpha = tf.stop_gradient(alpha)
            ciou_loss = diou_loss - alpha*v
            return tf.reduce_sum((1 - ciou_loss))
        return loss
class ClassificationLoss():
    @staticmethod
    def bce(label_smooth):
        def loss(y_true, y_pred):
            object_mask = y_true[..., 4]
            pos_num = tf.reduce_sum(object_mask, axis=[1,2,3])
            pos_num = tf.maximum(pos_num, 1)

            # y_true = y_true[..., 5:]
            # y_pred = y_pred[..., 5:]

            # classification_loss = object_mask*tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=args.label_smooth,reduction='none')(y_true[..., 5:], y_pred[..., 5:])
            classification_loss = object_mask * tf.keras.losses.binary_crossentropy(y_true[..., 5:],y_pred[..., 5:],from_logits=True,label_smoothing=label_smooth)

            return tf.reduce_sum(classification_loss, axis=[1,2,3])/pos_num
        return loss

    @staticmethod
    def focal(focal_alpha,focal_gamma,label_smooth):
        def loss(y_true, y_pred):
            # object_mask = y_true[..., 4]
            # y_true = y_true[..., 5:]
            # y_pred = y_pred[..., 5:]
            y_pred_sigmoid = tf.math.sigmoid(y_pred)
            alpha_weight = tf.where(y_true != 0, focal_alpha, 1.-focal_alpha)
            focal_weight = tf.where(y_true != 0, 1. - y_pred_sigmoid, y_pred_sigmoid)
            focal_weight = alpha_weight * focal_weight ** focal_gamma
            y_true = y_true * (1.0 - label_smooth) + 0.5 * label_smooth
            bce_loss = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred)
            focal_loss = focal_weight*bce_loss
            focal_loss = tf.reduce_sum(focal_loss)
            # focal_loss = tf.reduce_sum(focal_loss, axis=[-1])
            # focal_loss = tf.reduce_sum(object_mask*focal_loss,axis=[1, 2, 3])
            # normalizer = tf.maximum(tf.reduce_sum(object_mask, axis=[1, 2, 3]), 1.0)
            # focal_loss = tf.reduce_sum(focal_loss,axis=[1, 2, 3])
            # normalizer = tf.maximum(tf.reduce_sum(object_mask, axis=[1, 2, 3]), 1.0)
            # return focal_loss/normalizer
            return focal_loss
        return loss
import numpy as np
class ObjectLoss():
    @staticmethod
    def bce():
        def loss(y_true, y_pred, iou_score):

            object_loss_mask = y_true[..., 4]
            # object_loss_ori = tf.keras.losses.binary_crossentropy(y_true[..., 4:5], y_pred[..., 4:5], from_logits=True)
            iou_score = tf.maximum(iou_score, 0)
            iou_score = tf.stop_gradient(iou_score)
            object_loss_ori = tf.keras.losses.binary_crossentropy(tf.expand_dims(iou_score,axis=-1),
                                                                  y_pred[..., 4:5], from_logits=True)
            return tf.reduce_mean(object_loss_ori, axis=[1, 2, 3])

            # grid_height = tf.shape(y_true)[1]
            # grid_width = tf.shape(y_true)[2]
            # grid_xy = tf.stack(tf.meshgrid(tf.range(grid_width), tf.range(grid_height)), axis=-1)
            # grid_xy = tf.reshape(grid_xy, [tf.shape(grid_xy)[0], tf.shape(grid_xy)[1], 1, tf.shape(grid_xy)[2]])
            # grid_xy = tf.cast(grid_xy, tf.dtypes.float32)
            # y_true_cxcy = grid_xy+y_true[...,0:2]
            # y_true_x1y1x2y2 = tf.concat([y_true_cxcy - y_true[..., 2:4] / 2,y_true_cxcy + y_true[..., 2:4] / 2],axis=-1)
            # scales_x_y = tf.cast(args.scales_x_y[grid_index], tf.dtypes.float32)
            # scaled_pred_xy = tf.sigmoid(y_pred[..., 0:2]) * scales_x_y - 0.5 * (scales_x_y - 1)
            # pred_xy = (grid_xy + scaled_pred_xy)
            # normalized_anchors = tf.cast(yolo_anchors[args.model_type][grid_index], tf.dtypes.float32) / tf.cast(
            #     8 * (2 ** grid_index), tf.dtypes.float32)
            # pred_wh = (tf.sigmoid(y_pred[..., 2:4]) * 2) ** 2 * normalized_anchors
            # pred_x1y1x2y2 = tf.concat([pred_xy-pred_wh/2, pred_xy+pred_wh/2], axis=-1)
            #
            # iou_scores = broadcast_iou(pred_x1y1x2y2, tf.boolean_mask(y_true_x1y1x2y2, object_loss_mask))
            # iou_max_value = tf.reduce_max(iou_scores, axis=-1)
            # object_loss_ignore_mask = tf.cast(iou_max_value < args.ignore_thr, tf.dtypes.float32)
            # object_loss = object_loss_mask*object_loss_ori+(1-object_loss_mask)*object_loss_ignore_mask*object_loss_ori
            # num_valid = tf.reduce_sum(object_loss_mask,axis=[1,2,3])+tf.reduce_sum((1-object_loss_mask)*object_loss_ignore_mask,axis=[1,2,3])
            # num_valid = tf.maximum(num_valid,1)
            # return tf.reduce_sum(object_loss, axis=[1,2,3])/num_valid
        return loss



