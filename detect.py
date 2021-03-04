
"""A simple example on how to use model for inference."""
import os
import cv2
import numpy as np
import argparse
import random
import albumentations as A
from utils.nms import NonMaxSuppression
import tensorflow as tf
import logging
logging.getLogger().setLevel(logging.ERROR)

def parse_args(args):
    parser = argparse.ArgumentParser("test model")
    parser.add_argument('--model-dir', default='./export/best_model_d0_189_0.798/1')
    parser.add_argument('--class-names', default='./dataset/pothole.names')
    parser.add_argument('--tta', default=True)
    parser.add_argument('--score-threshold', default=0.1,type=float)
    parser.add_argument('--pic-dir', default='./dataset/pothole_voc/dataset_1/JPEGImages')
    return parser.parse_args(args)

def detect_batch_img(img,model):
    boxes, scores, classes, valid_detections = model(tf.convert_to_tensor(img,dtype=tf.uint8))
    return boxes, scores, classes, valid_detections
def tta_nms(boxes,scores,classes,valid_detections,args):
    all_boxes = []
    all_scores = []
    all_classes = []
    batch_index = 0
    valid_boxes = boxes[batch_index][0:valid_detections[batch_index]]
    valid_boxes[:, (0, 2)] = (1.-valid_boxes[:,(2,0)])
    all_boxes.append(valid_boxes)
    all_scores.append(scores[batch_index][0:valid_detections[batch_index]])
    all_classes.append(classes[batch_index][0:valid_detections[batch_index]])
    for batch_index in range(1,boxes.shape[0]):
        all_boxes.append(boxes[batch_index][0:valid_detections[batch_index]])
        all_scores.append(scores[batch_index][0:valid_detections[batch_index]])
        all_classes.append(classes[batch_index][0:valid_detections[batch_index]])
    all_boxes = np.concatenate(all_boxes,axis=0)
    all_scores = np.concatenate(all_scores, axis=0)
    all_classes = np.concatenate(all_classes, axis=0)
    all_boxes,all_scores,all_classes = np.array(all_boxes), np.array(all_scores), np.array(all_classes)
    boxes, scores, classes, valid_detections = NonMaxSuppression.diou_nms_np_tta(np.expand_dims(all_boxes,0),np.expand_dims(all_scores,0),np.expand_dims(all_classes,0),nms_score_threshold=args.score_threshold)
    boxes, scores, classes, valid_detections = np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes), np.squeeze(valid_detections)
    return boxes[:valid_detections], scores[:valid_detections], classes[:valid_detections]

def plot_one_box(img, box, color=None, label=None, line_thickness=None):
    """plot one box on image."""
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 7, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 7, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def plot_boxes(img,boxes,scores,classes,class_names,args):
    """plot all boxes on image."""
    for i in range(len(boxes)):
            x1y1 = boxes[i][0:2][::-1]
            x2y2 = boxes[i][2:4][::-1]
            plot_one_box(img,[x1y1[0],x1y1[1],x2y2[0],x2y2[1]],(255,0,255),label=str("%s:%0.2f" %(class_names[classes[i]],scores[i])))
    return img
def get_tta_tranform():
    """create test time augmentation for image ."""
    out_list=[]
    tta_transform = A.Compose([
        A.HorizontalFlip(p=1.),
        A.RandomBrightnessContrast(p=0.2),
    ])
    out_list.append(tta_transform)
    tta_transform = A.Compose([
        A.RandomBrightnessContrast(p=0.2),
    ])
    out_list.append(tta_transform)
    return out_list

def main(args):
    #load model
    model = tf.saved_model.load(args.model_dir)
    #read class labels
    with open(args.class_names) as f:
        class_names = f.read().splitlines()
    img_list = os.listdir(args.pic_dir)
    for img_name in img_list:
        img = cv2.imread(os.path.join(args.pic_dir, img_name))
        img_copy = img.copy()
        aug_imgs = []
        if args.tta:
            tta_transforms= get_tta_tranform()
            aug_imgs.append(tta_transforms[0](image=img_copy)['image'])
            aug_imgs.append(tta_transforms[1](image=img_copy)['image'])
        aug_imgs.append(img_copy)
        #predict model
        boxes,scores,classes,valid_detections = detect_batch_img(aug_imgs, model)
        boxes = boxes.numpy()
        scores = scores.numpy()
        classes = classes.numpy()
        valid_detections = valid_detections.numpy()
        #Visualize results
        boxes, scores, classes = tta_nms(boxes, scores, classes,valid_detections,args)
        plot_boxes(img_copy,boxes,scores,classes,class_names,args)
        cv2.imshow("demo", img_copy/255)
        cv2.waitKey(0)

import sys
if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)
