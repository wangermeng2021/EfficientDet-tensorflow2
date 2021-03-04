
import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from labeler.labeler_builder import get_labels
# from data_augment import cutmix
from generator import data_augment
# from get_y_true import get_y_true_with_multi_class,get_y_true_with_one_class
from utils import preprocess

# from data_augment import np_random_color_distort
# from utils import  aug_gluoncv
from pycocotools.coco import COCO

from utils import aug_gluoncv
import copy
from utils.preprocess import resize_img,resize_img_aug
from config import efficientdet_config
class Generator(tf.keras.utils.Sequence):

    def __init__(self, args,mode):
        self.args = copy.deepcopy(args)
        self.mode = mode
        self.img_path_list,self.boxes_and_labels = self.get_data(self.args, mode=mode)
        self.img_path_list,self.boxes_and_labels, self.data_index = self.pad_dataset(self.img_path_list,self.boxes_and_labels)
        self.get_labels_fun = get_labels(args)

        model_args = efficientdet_config.get_struct_args(args)
        self.img_size = model_args.image_size
    def pad_dataset(self,img_path_list,boxes_and_labels):
        if  self.mode == 'train':
            if self.args.augment == 'mosaic':
                self.args.batch_size *= 4
            pad_num = self.args.batch_size-len(img_path_list) % self.args.batch_size
            for _ in range(pad_num):
                pi = np.random.choice(range(len(img_path_list)))
                img_path_list.append(img_path_list[pi])
                boxes_and_labels.append(copy.deepcopy(boxes_and_labels[pi]))
            self.resize_fun = resize_img_aug
            self.augment = None
        else:
            self.resize_fun = resize_img
            self.args.augment = None
            self.augment = None

        data_index = np.empty([len(img_path_list)], np.int32)
        for index in range(len(img_path_list)):
            data_index[index] = index
        if self.mode == 'train':
            np.random.shuffle(data_index)
        return img_path_list,boxes_and_labels,data_index
    def get_classes_num(self):
        return int(self.args.num_classes)
    def get_size(self):
        return len(self.img_path_list)
    def __len__(self):
        if self.mode == 'train':
            return len(self.img_path_list) // self.args.batch_size
        else:
            return int(np.ceil(len(self.img_path_list) / self.args.batch_size))
    def on_epoch_end(self):
        if self.mode == 'train':
            np.random.shuffle(self.data_index)
    def __getitem__(self, item):


        with tf.device("/cpu:0"):
            groundtruth_valids = np.zeros([self.args.batch_size],np.int)

            # random_img_size = np.random.choice(self.args.multi_scale)
            random_img_size = self.img_size

            self.gluoncv_aug = aug_gluoncv.YOLO3DefaultTrainTransform(random_img_size, random_img_size)
            batch_img = np.zeros([self.args.batch_size, random_img_size, random_img_size, 3])
            batch_boxes = np.empty([self.args.batch_size, self.args.max_box_num_per_image, 5])
            batch_boxes_list = []
            for batch_index, file_index in enumerate(self.data_index[item*self.args.batch_size:(item+1)*self.args.batch_size]):
                #get image from file
                img_path = self.img_path_list[file_index]
                img = self.read_img(img_path)
                img, scale, pad = self.resize_fun(img, (random_img_size, random_img_size))
                batch_img[batch_index, 0:img.shape[0], 0:img.shape[1], :] = img
                boxes = self.boxes_and_labels[file_index]
                boxes = copy.deepcopy(boxes)
                boxes[:, 0:4] *= scale
                half_pad = pad // 2
                boxes[:, 0:4] += np.tile(half_pad, 2)
                batch_boxes_list.append(boxes)
                groundtruth_valids[batch_index] = boxes.shape[0]
                boxes = np.pad(boxes, [(0, self.args.max_box_num_per_image-boxes.shape[0]), (0, 0)], mode='constant')
                batch_boxes[batch_index] = boxes
            tail_batch_size = len(batch_boxes_list)
            #augment
            if self.args.augment == 'mosaic':
                new_batch_size = self.args.batch_size//4
                for bi in range(new_batch_size):
                    four_img, four_boxes, one_img, one_boxes = data_augment.load_mosaic(batch_img[bi * 4:(bi + 1) * 4],
                                                                                        batch_boxes_list[bi * 4:(bi + 1) * 4])
                    data_augment.random_hsv(one_img)
                    data_augment.random_left_right_flip(one_img, one_boxes)
                    groundtruth_valids[bi] = one_boxes.shape[0]
                    one_boxes = np.pad(one_boxes,[(0, self.args.max_box_num_per_image-one_boxes.shape[0]), (0, 0)], mode='constant')
                    batch_img[bi] = one_img
                    batch_boxes[bi] = one_boxes

                batch_img = batch_img[0:new_batch_size]
                batch_boxes = batch_boxes[0:new_batch_size]
            elif self.args.augment == 'only_flip_left_right':
                for bi in range(self.args.batch_size):
                    data_augment.random_left_right_flip(batch_img[bi], batch_boxes[bi])
            elif self.args.augment == 'ssd_random_crop':
                batch_img = batch_img.astype(np.uint8)
                for di in range(self.args.batch_size):
                    batch_img[di], batch_boxes_list[di] = self.gluoncv_aug(batch_img[di], batch_boxes_list[di])
                    batch_boxes[di] = np.pad(batch_boxes_list[di], [(0, self.args.max_box_num_per_image - batch_boxes_list[di].shape[0]), (0, 0)])
                    groundtruth_valids[di] = batch_boxes_list[di].shape[0]

            batch_img = batch_img[0:tail_batch_size]
            batch_boxes = batch_boxes[0:tail_batch_size]
            groundtruth_valids = groundtruth_valids[0:tail_batch_size]
            ###############
            batch_img = preprocess.normalize(batch_img)
            batch_boxes[..., 0:4] /= np.tile(batch_img.shape[1:3][::-1], [2])
            batch_img = batch_img.astype(np.float32)
            batch_boxes = batch_boxes.astype(np.float32)
            ###############

            # batch_boxes = np.array([[[0.2, 0.1, 0.5, 0.5, 3], [0.5, 0.3, 0.8, 0.9, 3]]])
            # groundtruth_valids = np.array([2])
            # y_true = self.get_labels_fun.get_labels(random_img_size, batch_boxes, groundtruth_valids)
            # batch_boxes = np.array([[[0.2, 0.1, 0.5, 0.5, 3], [0.5, 0.3, 0.8, 0.9, 3]]])
            # groundtruth_valids = np.array([2])
            # y_true1 = get_y_true(416, batch_boxes, groundtruth_valids, args)
            # print(np.all(y_true[0]==y_true1[0]))
            # print(np.all(y_true[1] == y_true1[1]))

            if self.mode == 'pred':
                return batch_img, batch_boxes, groundtruth_valids
            y_true = self.get_labels_fun.get_labels(random_img_size, batch_boxes, groundtruth_valids)
            return batch_img, y_true
            return batch_img, y_true, batch_boxes

    def read_img(self, path):
        image = np.ascontiguousarray(Image.open(path).convert('RGB'))
        return image[:, :, ::-1]




