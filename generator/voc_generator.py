
import os
import time
import warnings
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

# from data_augment import cutmix
# from generator import data_augment
# from get_y_true import get_y_true,get_y_true_with_one_class
"""Train YOLOv3 with random shapes."""
import argparse
import os
import logging
import time
import warnings
import numpy as np

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

from generator.generator import Generator
class VocGenerator(Generator):

    def __init__(self, args, mode='train'):
        super(VocGenerator, self).__init__(args,mode)
    def get_data(self,args, mode):

        if mode == 'train':
            sets = args.voc_train_set
        else:
            sets = args.voc_val_set
        class_names_dict={}
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),args.class_names)) as f:
            class_names = f.read().splitlines()
            for i in range(len(class_names)):
                class_names_dict[class_names[i]] = i
        root_dir = args.dataset
        skip_difficult = int(args.voc_skip_difficult)

        xml_path_list = []
        img_path_list = []

        sets_list = sets.split(',')
        sets_name = []
        for si in range(len(sets_list)//2):
            sets_name.append((sets_list[si * 2], sets_list[si * 2+1]))

        for voc_year, voc_set in sets_name:
            txt_path = os.path.join(root_dir, str(voc_year), 'ImageSets', 'Main', voc_set+'.txt')
            with open(txt_path) as f:
                lines = f.readlines()
            for line in lines:
                valid_label = self.check_img_and_xml(os.path.join(root_dir,  str(voc_year), 'JPEGImages', line.strip() + '.jpg'),os.path.join(root_dir, str(voc_year), 'Annotations', line.strip()+'.xml'))
                if valid_label:
                    xml_path_list.append(os.path.join(root_dir, str(voc_year), 'Annotations', line.strip()+'.xml'))
                    img_path_list.append(os.path.join(root_dir, str(voc_year), 'JPEGImages', line.strip() + '.jpg'))

        boxes_and_labels = []
        for xml_path in xml_path_list:
            boxes_and_labels.append(self.parse_xml(xml_path,skip_difficult,class_names_dict))
        return img_path_list,boxes_and_labels

    def check_img_and_xml(self, img_path, xml_path):
        try:
            tree = ET.parse(xml_path)
            xml_root = tree.getroot()
            num_valid_boxes = 0
            for element in xml_root.iter('object'):
                # truncated = int(element.find('truncated').text)
                difficult = int(element.find('difficult').text)
                if difficult:
                    continue
                num_valid_boxes += 1
            if num_valid_boxes == 0:
                return False
        except:
            return False
        return True

    def parse_xml(self,file_path,skip_difficult,class_names_dict):
        try:
            tree = ET.parse(file_path)
            xml_root = tree.getroot()

            size = xml_root.find('size')
            width = float(size.find('width').text)
            height = float(size.find('height').text)

            boxes = np.empty((len(xml_root.findall('object')), 5))
            box_index = 0
            for i, element in enumerate(xml_root.iter('object')):
                # truncated = int(element.find('truncated').text)
                difficult = int(element.find('difficult').text)
                class_name = element.find('name').text
                box = np.zeros((4,))
                label = class_names_dict[class_name]
                bndbox = element.find('bndbox')

                box[0] = float(bndbox.find('xmin').text)-1
                box[1] = float(bndbox.find('ymin').text)-1
                box[2] = float(bndbox.find('xmax').text)-1
                box[3] = float(bndbox.find('ymax').text)-1

                # assert 0 <= box[0] < width
                # assert 0 <= box[1] < height
                # assert box[0] < box[2] < width
                # assert box[1] < box[3] < height
                box[0] = np.maximum(box[0], 0)
                box[1] = np.maximum(box[1], 0)
                box[2] = np.minimum(box[2], width-1)
                box[3] = np.minimum(box[3], height-1)

                # if truncated and self.skip_truncated:
                #     continue
                if difficult and skip_difficult:
                    continue

                boxes[box_index, 0:4] = box
                boxes[box_index, 4] = int(label)
                box_index += 1
            return boxes[0:box_index]
            # return boxes[boxes[...,3]>0]
        except ET.ParseError as e:
            ValueError('there is an error in parsing xml file: {}: {}'.format(file_path, e))

# from utils.common import cfg_to_struct
# from config.config import CFG as train_cfgs
# cfgs = cfg_to_struct(train_cfgs)
# for train_args, model_args in cfgs:
#     voc_generator = VocGenerator(train_args, model_args,'train')
#     ii = 0
#     # for imgs, boxes, _ in voc_generator:
#     for imgs, y_true, boxes in voc_generator:
#         for box1_index, box1 in enumerate(boxes):
#             for box2 in box1:
#                 if box2[2] == 0:
#                     continue
#                 box = box2[0:4] * np.tile(imgs[box1_index].shape[0:2][::-1], [2])
#                 box = box.astype(np.int32)
#                 cv2.rectangle(imgs[box1_index], (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 3)
#             # cv2.imshow('1', imgs[box1_index])
#             cv2.imwrite("s1/{}.jpg".format(ii), imgs[box1_index] * 255)
#             ii += 1
#             # cv2.waitKey(0)
#
# exit()

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#
# def parse_args(args):
#     parser = argparse.ArgumentParser(description='Simple training script for using ScaledYOLOv4.')
#     #save model
#     parser.add_argument('--output-model-dir', default='./output_model')
#     #training
#     parser.add_argument('--epochs', default=300, type=int)
#     parser.add_argument('--batch-size', default=64, type=int)
#     parser.add_argument('--start-eval-epoch', default=10, type=int)
#     parser.add_argument('--eval-epoch-interval', default=1)
#     #model
#     parser.add_argument('--model-type', default='tiny', help="choices=['tiny','p5','p6','p7']")
#     parser.add_argument('--use-pretrain', default=False, type=bool)
#     parser.add_argument('--tiny-coco-pretrained-weights',
#                         default='./pretrain/ScaledYOLOV4_tiny_coco_pretrain/coco_pretrain')
#     parser.add_argument('--p5-coco-pretrained-weights',
#                         default='./pretrain/ScaledYOLOV4_p5_coco_pretrain/coco_pretrain')
#     parser.add_argument('--p6-coco-pretrained-weights',
#                         default='./pretrain/ScaledYOLOV4_p6_coco_pretrain/coco_pretrain')
#     parser.add_argument('--checkpoints-dir', default='./checkpoints',help="Directory to store  checkpoints of model during training.")
#     #loss
#     parser.add_argument('--box-regression-loss', default='ciou',help="choices=['giou','diou','ciou']")
#     parser.add_argument('--classification-loss', default='bce', help="choices=['ce','bce','focal']")
#     parser.add_argument('--focal-alpha', default= 0.25)
#     parser.add_argument('--focal-gamma', default=2.0)
#     parser.add_argument('--ignore-thr', default=0.7)
#     parser.add_argument('--reg-losss-weight', default=0.05)
#     parser.add_argument('--obj-losss-weight', default=1.0)
#     parser.add_argument('--cls-losss-weight', default=0.5)
#     #dataset
#     parser.add_argument('--dataset-type', default='voc', help="voc,coco")
#     parser.add_argument('--num-classes', default=1)
#     parser.add_argument('--class-names', default='pothole.names', help="voc.names,coco.names")
#     parser.add_argument('--dataset', default='/home/wangem1/dataset/Pothole/pothole_voc')#
#     parser.add_argument('--voc-train-set', default='dataset_1,train')
#     parser.add_argument('--voc-val-set', default='dataset_1,val')
#     parser.add_argument('--voc-skip-difficult', default=True)
#     parser.add_argument('--coco-train-set', default='train2017')
#     parser.add_argument('--coco-valid-set', default='val2017')
#     '''
#     voc dataset directory:
#         VOC2007
#                 Annotations
#                 ImageSets
#                 JPEGImages
#         VOC2012
#                 Annotations
#                 ImageSets
#                 JPEGImages
#     coco dataset directory:
#         annotations/instances_train2017.json
#         annotations/instances_val2017.json
#         images/train2017
#         images/val2017
#     '''
#     parser.add_argument('--augment', default='mosaic',help="choices=[None,'only_flip_left_right','ssd_random_crop','mosaic']")
#     parser.add_argument('--multi-scale', default=[416],help="Input data shapes for training, use 320+32*i(i>=0)")#896
#     parser.add_argument('--max-box-num-per-image', default=100)
#     #optimizer
#     parser.add_argument('--optimizer', default='sgd', help="choices=[adam,sgd]")
#     parser.add_argument('--momentum', default=0.9)
#     parser.add_argument('--nesterov', default=True)
#     parser.add_argument('--weight-decay', default=5e-4)
#     #lr scheduler
#     parser.add_argument('--lr-scheduler', default='warmup_cosinedecay', type=str, help="choices=['step','warmup_cosinedecay']")
#     parser.add_argument('--init-lr', default=1e-3, type=float)
#     parser.add_argument('--lr-decay', default=0.1, type=float)
#     parser.add_argument('--lr-decay-epoch', default=[160, 180])
#     parser.add_argument('--warmup-epochs', default=10, type=int)
#     parser.add_argument('--warmup-lr', default=1e-6, type=float)
#     #postprocess
#     parser.add_argument('--nms', default='diou_nms', help="choices=['hard_nms','diou_nms']")
#     parser.add_argument('--nms-max-box-num', default=300)
#     parser.add_argument('--nms-iou-threshold', default=0.2, type=float)
#     parser.add_argument('--nms-score-threshold', default=0.01, type=float)
#     #anchor
#     parser.add_argument('--anchor-match-type', default='wh_ratio',help="choices=['iou','wh_ratio']")
#     parser.add_argument('--anchor-match-iou_thr', default=0.2, type=float)
#     parser.add_argument('--anchor-match-wh-ratio-thr', default=4.0, type=float)
#
#     parser.add_argument('--label-smooth', default=0.0, type=float)
#     parser.add_argument('--scales-x-y', default=[2., 2., 2., 2., 2.])
#     parser.add_argument('--accumulated-gradient-num', default=1, type=int)
#
#     parser.add_argument('--tensorboard', default=True, type=bool)
#
#     return parser.parse_args(args)
# import argparse
# import sys
#
#
#
# if __name__ == "__main__":
#     args = parse_args(sys.argv[1:])
#     voc_generator = VocGenerator(args, mode=0)
#
#     for imgs, y_true,boxes in voc_generator:
#         for box1_index, box1 in enumerate(boxes):
#             for box2 in box1:
#                 if box2[2] == 0:
#                     continue
#                 box = box2[0:4] * np.tile(imgs[box1_index].shape[0:2][::-1], [2])
#                 box = box.astype(np.int32)
#                 cv2.rectangle(imgs[box1_index],(box[0],box[1]),(box[2],box[3]),(255,0,0),3)
#             cv2.imshow('1', imgs[box1_index])
#             cv2.waitKey(0)
