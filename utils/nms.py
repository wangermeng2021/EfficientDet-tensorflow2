
"""

"""
import numpy as np
import tensorflow as tf
class NonMaxSuppression():

    @staticmethod
    def hard_nms_tf(boxes, scores, nms_max_box_num,nms_iou_threshold,nms_score_threshold):
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=scores,
            max_output_size_per_class=nms_max_box_num,
            max_total_size=nms_max_box_num,
            iou_threshold=nms_iou_threshold,
            nms_score_threshold=nms_score_threshold
        )
        return boxes, scores, classes, valid_detections
    @staticmethod
    def soft_nms(boxes, iou_thr, score_thr=0.0):
        pass
    @staticmethod
    def hard_nms_np(batch_boxes, batch_scores,  nms_max_box_num,nms_iou_threshold,nms_score_threshold):
        """Implementing  diou non-maximum suppression in numpy
         Args:
           batch_boxes: detection boxes with shape (N, num, 4) and box format is [x1, y1, x2, y2].
           batch_scores:detection scores with shape (N, num_class).
         Returns:
            a list of numpy array: [boxes, scores, classes, num_valid].
         """



        batch_classes = np.argmax(batch_scores, axis=-1)
        batch_scores = np.max(batch_scores, axis=-1)

        batch_size = np.shape(batch_boxes)[0]

        batch_result_boxes = np.empty([batch_size, nms_max_box_num, 4])
        batch_result_scores = np.empty([batch_size, nms_max_box_num])
        batch_result_classes = np.empty([batch_size, nms_max_box_num],dtype=np.int32)
        batch_result_valid = np.empty([batch_size],dtype=np.int32)

        for batch_index in range(batch_size):
            # print(batch_result_boxes[0])
            boxes = batch_boxes[batch_index]
            scores = batch_scores[batch_index]

            classes = batch_classes[batch_index]

            valid_mask = scores > nms_score_threshold

            if np.sum(valid_mask) == 0:
                batch_result_boxes[batch_index] = np.zeros([nms_max_box_num, 4])
                batch_result_scores[batch_index] = np.zeros([nms_max_box_num])
                batch_result_classes[batch_index] = np.zeros([nms_max_box_num])
                batch_result_valid[batch_index] = 0
                continue

            boxes = boxes[valid_mask]
            scores = scores[valid_mask]
            classes = classes[valid_mask]

            sorted_index = np.argsort(scores)[::-1]
            boxes = boxes[sorted_index]
            scores = scores[sorted_index]
            classes = classes[sorted_index]

            result_boxes = []
            result_scores = []
            result_classes = []
            while boxes.shape[0] > 0:
                result_boxes.append(boxes[0])
                result_scores.append(scores[0])
                result_classes.append(classes[0])
                inter_wh = np.maximum(
                    np.minimum(boxes[0, 2:4], boxes[1:, 2:4]) - np.maximum(boxes[0, 0:2], boxes[1:, 0:2]), 0)
                inter_area = inter_wh[:, 0] * inter_wh[:, 1]

                box1_wh = boxes[0, 2:4] - boxes[0, 0:2]
                box2_wh = boxes[1:, 2:4] - boxes[1:, 0:2]

                iou_score = inter_area / (box1_wh[0] * box1_wh[1] + box2_wh[:, 0] * box2_wh[:, 1] - inter_area + 1e-7)
                # center_dist = np.sum(
                #     np.square((boxes[0, 2:4] + boxes[0, 0:2]) / 2 - (boxes[1:, 2:4] + boxes[1:, 0:2]) / 2),
                #     axis=-1)
                # bounding_rect_wh = np.maximum(boxes[0, 2:4], boxes[1:, 2:4]) - np.minimum(boxes[0, 0:2], boxes[1:, 0:2])
                # diagonal_dist = np.sum(np.square(bounding_rect_wh), axis=-1)
                # diou = iou_score - center_dist / diagonal_dist
                diou = iou_score
                valid_mask = diou <= nms_iou_threshold
                boxes = boxes[1:][valid_mask]
                scores = scores[1:][valid_mask]
                classes = classes[1:][valid_mask]

            # result_boxes = np.array(result_boxes)
            # result_scores = np.array(result_scores)
            # result_classes = np.array(result_classes)
            num_valid = len(result_boxes)
            num_valid = np.minimum(num_valid, nms_max_box_num)
            result_boxes = np.array(result_boxes)[:num_valid,:]
            result_scores = np.array(result_scores)[:num_valid]
            result_classes = np.array(result_classes)[:num_valid]
            pad_size = nms_max_box_num - num_valid
            result_boxes = np.pad(result_boxes, ((0, pad_size), (0, 0)))
            result_scores = np.pad(result_scores, ((0, pad_size),))
            result_classes = np.pad(result_classes, ((0, pad_size),))


            batch_result_boxes[batch_index] = result_boxes
            batch_result_scores[batch_index] = result_scores
            batch_result_classes[batch_index] = result_classes
            batch_result_valid[batch_index] = num_valid

        return batch_result_boxes, batch_result_scores, batch_result_classes, batch_result_valid
    @staticmethod
    def diou_nms_np(batch_boxes, batch_scores,  nms_max_box_num,nms_iou_threshold,nms_score_threshold):
        """Implementing  diou non-maximum suppression in numpy
         Args:
           batch_boxes: detection boxes with shape (N, num, 4) and box format is [x1, y1, x2, y2].
           batch_scores:detection scores with shape (N, num_class).
         Returns:
            a list of numpy array: [boxes, scores, classes, num_valid].
         """


        batch_classes = np.argmax(batch_scores, axis=-1)
        batch_scores = np.max(batch_scores, axis=-1)

        batch_size = np.shape(batch_boxes)[0]

        batch_result_boxes = np.empty([batch_size, nms_max_box_num, 4])
        batch_result_scores = np.empty([batch_size, nms_max_box_num])
        batch_result_classes = np.empty([batch_size, nms_max_box_num],dtype=np.int32)
        batch_result_valid = np.empty([batch_size],dtype=np.int32)

        for batch_index in range(batch_size):
            # print(batch_result_boxes[0])
            boxes = batch_boxes[batch_index]
            scores = batch_scores[batch_index]
            classes = batch_classes[batch_index]

            valid_mask = scores > nms_score_threshold

            if np.sum(valid_mask) == 0:
                batch_result_boxes[batch_index] = np.zeros([nms_max_box_num, 4])
                batch_result_scores[batch_index] = np.zeros([nms_max_box_num])
                batch_result_classes[batch_index] = np.zeros([nms_max_box_num])
                batch_result_valid[batch_index] = 0
                continue

            boxes = boxes[valid_mask]
            scores = scores[valid_mask]
            classes = classes[valid_mask]

            sorted_index = np.argsort(scores)[::-1]
            boxes = boxes[sorted_index]
            scores = scores[sorted_index]
            classes = classes[sorted_index]

            result_boxes = []
            result_scores = []
            result_classes = []
            while boxes.shape[0] > 0:
                result_boxes.append(boxes[0])
                result_scores.append(scores[0])
                result_classes.append(classes[0])
                inter_wh = np.maximum(
                    np.minimum(boxes[0, 2:4], boxes[1:, 2:4]) - np.maximum(boxes[0, 0:2], boxes[1:, 0:2]), 0)
                inter_area = inter_wh[:, 0] * inter_wh[:, 1]

                box1_wh = boxes[0, 2:4] - boxes[0, 0:2]
                box2_wh = boxes[1:, 2:4] - boxes[1:, 0:2]

                iou_score = inter_area / (box1_wh[0] * box1_wh[1] + box2_wh[:, 0] * box2_wh[:, 1] - inter_area + 1e-7)
                center_dist = np.sum(
                    np.square((boxes[0, 2:4] + boxes[0, 0:2]) / 2 - (boxes[1:, 2:4] + boxes[1:, 0:2]) / 2),
                    axis=-1)
                bounding_rect_wh = np.maximum(boxes[0, 2:4], boxes[1:, 2:4]) - np.minimum(boxes[0, 0:2], boxes[1:, 0:2])
                diagonal_dist = np.sum(np.square(bounding_rect_wh), axis=-1)
                diou = iou_score - center_dist / (diagonal_dist+1e-7)
                # print(diou)
                valid_mask = diou <= nms_iou_threshold
                boxes = boxes[1:][valid_mask]
                scores = scores[1:][valid_mask]
                classes = classes[1:][valid_mask]

            num_valid = len(result_boxes)
            num_valid = np.minimum(num_valid, nms_max_box_num)
            result_boxes = np.array(result_boxes)[:num_valid,:]
            result_scores = np.array(result_scores)[:num_valid]
            result_classes = np.array(result_classes)[:num_valid]
            pad_size = nms_max_box_num - num_valid
            result_boxes = np.pad(result_boxes, ((0, pad_size), (0, 0)))
            result_scores = np.pad(result_scores, ((0, pad_size),))
            result_classes = np.pad(result_classes, ((0, pad_size),))

            batch_result_boxes[batch_index] = result_boxes
            batch_result_scores[batch_index] = result_scores
            batch_result_classes[batch_index] = result_classes
            batch_result_valid[batch_index] = num_valid

        return batch_result_boxes, batch_result_scores, batch_result_classes, batch_result_valid
    @staticmethod
    def diou_nms_np_tta(batch_boxes, batch_scores, batch_classes,nms_max_box_num=300,nms_iou_threshold=0.2,nms_score_threshold=0.1):
        """Implementing  diou non-maximum suppression in numpy
         Args:
           batch_boxes: detection boxes with shape (N, num, 4) and box format is [x1, y1, x2, y2].
           batch_scores:detection scores with shape (N, num_class).
         Returns:
            a list of numpy array: [boxes, scores, classes, num_valid].
         """

        # batch_classes = np.argmax(batch_scores, axis=-1)
        # batch_scores = np.max(batch_scores, axis=-1)

        batch_size = np.shape(batch_boxes)[0]

        batch_result_boxes = np.empty([batch_size, nms_max_box_num, 4])
        batch_result_scores = np.empty([batch_size, nms_max_box_num])
        batch_result_classes = np.empty([batch_size, nms_max_box_num],dtype=np.int32)
        batch_result_valid = np.empty([batch_size],dtype=np.int32)

        for batch_index in range(batch_size):

            boxes = batch_boxes[batch_index]
            scores = batch_scores[batch_index]

            classes = batch_classes[batch_index]

            valid_mask = scores > nms_score_threshold

            if np.sum(valid_mask) == 0:
                batch_result_boxes[batch_index] = np.zeros([nms_max_box_num, 4])
                batch_result_scores[batch_index] = np.zeros([nms_max_box_num])
                batch_result_classes[batch_index] = np.zeros([nms_max_box_num])
                batch_result_valid[batch_index] = 0
                continue

            boxes = boxes[valid_mask]
            scores = scores[valid_mask]
            classes = classes[valid_mask]

            sorted_index = np.argsort(scores)[::-1]
            boxes = boxes[sorted_index]
            scores = scores[sorted_index]
            classes = classes[sorted_index]

            result_boxes = []
            result_scores = []
            result_classes = []
            while boxes.shape[0] > 0:
                result_boxes.append(boxes[0])
                result_scores.append(scores[0])
                result_classes.append(classes[0])
                inter_wh = np.maximum(
                    np.minimum(boxes[0, 2:4], boxes[1:, 2:4]) - np.maximum(boxes[0, 0:2], boxes[1:, 0:2]), 0)
                inter_area = inter_wh[:, 0] * inter_wh[:, 1]

                box1_wh = boxes[0, 2:4] - boxes[0, 0:2]
                box2_wh = boxes[1:, 2:4] - boxes[1:, 0:2]

                iou_score = inter_area / (box1_wh[0] * box1_wh[1] + box2_wh[:, 0] * box2_wh[:, 1] - inter_area + 1e-7)
                center_dist = np.sum(
                    np.square((boxes[0, 2:4] + boxes[0, 0:2]) / 2 - (boxes[1:, 2:4] + boxes[1:, 0:2]) / 2),
                    axis=-1)
                bounding_rect_wh = np.maximum(boxes[0, 2:4], boxes[1:, 2:4]) - np.minimum(boxes[0, 0:2], boxes[1:, 0:2])
                diagonal_dist = np.sum(np.square(bounding_rect_wh), axis=-1)
                diou = iou_score - center_dist / (diagonal_dist+1e-7)
                valid_mask = diou <= nms_iou_threshold
                boxes = boxes[1:][valid_mask]
                scores = scores[1:][valid_mask]
                classes = classes[1:][valid_mask]

            num_valid = len(result_boxes)
            num_valid = np.minimum(num_valid, nms_max_box_num)
            result_boxes = np.array(result_boxes)[:num_valid,:]
            result_scores = np.array(result_scores)[:num_valid]
            result_classes = np.array(result_classes)[:num_valid]
            pad_size = nms_max_box_num - num_valid
            result_boxes = np.pad(result_boxes, ((0, pad_size), (0, 0)))
            result_scores = np.pad(result_scores, ((0, pad_size),))
            result_classes = np.pad(result_classes, ((0, pad_size),))

            batch_result_boxes[batch_index] = result_boxes
            batch_result_scores[batch_index] = result_scores
            batch_result_classes[batch_index] = result_classes
            batch_result_valid[batch_index] = num_valid




        return batch_result_boxes, batch_result_scores, batch_result_classes, batch_result_valid
def yolov4_nms(train_args):
    if train_args.postprocess.nms == 'hard_nms':
        return NonMaxSuppression.hard_nms_np
    elif train_args.postprocess.nms == 'diou_nms':
        return NonMaxSuppression.diou_nms_np
    else:
        raise ValueError('{} is not supported!'.format(train_args.postprocess.nms))

    pass



