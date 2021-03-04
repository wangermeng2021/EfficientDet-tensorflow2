
import numpy as np
from model.efficientdet import anchors
from config import efficientdet_config
class EfficientdetLabeler():
    def __init__(self,args):
        self.model_args = efficientdet_config.get_struct_args(args)
        input_anchors = anchors.Anchors(self.model_args.min_level, self.model_args.max_level, self.model_args.num_scales,
                                        self.model_args.aspect_ratios, self.model_args.anchor_scale,
                                        self.model_args.image_size)
        self.anchor_labeler = anchors.AnchorLabeler(input_anchors, self.model_args.num_classes)

    def get_labels(self, img_size, batch_boxes, groundtruth_valids):
        #
        batch_boxes[...,0:4]*=img_size
        #convert x1y1x2y2 to y1x1y2x2
        batch_boxes_temp = batch_boxes[...,[1,0]]
        batch_boxes[..., 0:2] = batch_boxes_temp
        batch_boxes_temp = batch_boxes[...,[3,2]]
        batch_boxes[..., 2:4] = batch_boxes_temp

        batch_boxes[...,4]+=1.

        num_level = self.model_args.max_level - self.model_args.min_level + 1
        batch_level_cls_targets = [[] for _ in range(num_level)]
        batch_level_box_targets = [[] for _ in range(num_level)]

        for bi in range(batch_boxes.shape[0]):
            batch_boxes_valid = batch_boxes[bi][0:groundtruth_valids[bi]]

            (cls_targets, box_targets) = self.anchor_labeler.label_anchors(batch_boxes_valid[:,0:4], batch_boxes_valid[:,4:5])

            for level in range(self.model_args.min_level, self.model_args.max_level + 1):

                batch_level_cls_targets[level-self.model_args.min_level].append(cls_targets[level].numpy())
                batch_level_box_targets[level-self.model_args.min_level].append(box_targets[level].numpy())
        output_cls_targets = []
        output_box_targets = []
        for level in range(num_level):
            output_cls_targets.append(np.array(batch_level_cls_targets[level]))
            output_box_targets.append(np.array(batch_level_box_targets[level]))
        return output_cls_targets, output_box_targets
        output_cls_targets = {}
        output_box_targets = {}
        for level in range(num_level):
            output_cls_targets['level_{}_cls_loss'.format(level)] = np.array(batch_level_cls_targets[level])
            output_cls_targets['level_{}_box_loss'.format(level)] = np.array(batch_level_box_targets[level])
        return output_cls_targets.update(output_box_targets)


        # output_cls_targets.extend(output_box_targets)
        # return tuple(output_cls_targets)