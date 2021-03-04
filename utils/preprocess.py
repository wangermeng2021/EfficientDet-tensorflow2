
import numpy as np
import cv2
import tensorflow as tf

def normalize(img, style=2):
    if style == 0:
        img = img / 255.
    elif style == 1:#tensorflow
        img = img / 127.5 - 1.0
    else:
        img = img / 255.
        img = img - [0.485, 0.456, 0.406]
        img = img / [0.229, 0.224, 0.225]
    return img

def resize_img_aug(img,dst_size):
    img_wh = img.shape[0:2][::-1]
    dst_size = np.array(dst_size)
    scale = dst_size/img_wh
    min_scale = np.min(scale)
    random_resize_style = np.random.randint(0, 5)
    resize_list = [cv2.INTER_AREA,cv2.INTER_CUBIC,cv2.INTER_LINEAR,cv2.INTER_NEAREST,cv2.INTER_LANCZOS4]
    img = cv2.resize(img, None, fx=min_scale, fy=min_scale, interpolation=resize_list[random_resize_style])
    img_wh = img.shape[0:2][::-1]
    pad_size = dst_size - img_wh
    half_pad_size = pad_size//2
    img = np.pad(img,[(half_pad_size[1],pad_size[1]-half_pad_size[1]),(half_pad_size[0],pad_size[0]-half_pad_size[0]),(0,0)], constant_values=np.random.randint(0, 255))
    return img, min_scale, pad_size

def resize_img(img,dst_size):
    img_wh = img.shape[0:2][::-1]
    dst_size = np.array(dst_size)
    scale = dst_size/img_wh
    min_scale = np.min(scale)
    img = cv2.resize(img, None, fx=min_scale, fy=min_scale)
    img_wh = img.shape[0:2][::-1]
    pad_size = dst_size - img_wh
    half_pad_size = pad_size//2
    img = np.pad(img,[(half_pad_size[1],pad_size[1]-half_pad_size[1]),(half_pad_size[0],pad_size[0]-half_pad_size[0]),(0,0)])
    return img, min_scale, pad_size

@tf.function()
def resize_img_tf(img,dst_size):
    img_hw = tf.shape(img)[1:3]
    img_hw = tf.cast(img_hw,tf.dtypes.float32)
    scale = dst_size/img_hw
    min_scale = tf.reduce_min(scale)
    scaled_img_hw = tf.cast(img_hw*min_scale,tf.dtypes.int32)
    scaled_image = tf.image.resize(img, [scaled_img_hw[0], scaled_img_hw[1]], method=tf.image.ResizeMethod.BILINEAR)
    pad_size = dst_size - scaled_img_hw
    half_pad_size = pad_size//2
    scaled_image = tf.pad(scaled_image,[(0,0),(half_pad_size[0],pad_size[0]-half_pad_size[0]),(half_pad_size[1],pad_size[1]-half_pad_size[1]),(0,0)])
    scaled_image = tf.image.pad_to_bounding_box(scaled_image, 0, 0,dst_size[0],dst_size[1])
    return scaled_image, min_scale, half_pad_size
