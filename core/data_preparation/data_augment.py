# coding=utf-8
#================================================================
#   Copyright (C) 2018 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : convert_tfrecord.py
#   Author      : tsing-cv
#   Created date: 2019-02-16 17:32:23
#   Description :
#
#================================================================
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
import cv2

def random_rot90(image, gtboxes_and_label):
    h, w = tf.cast(tf.shape(image)[0], tf.float32), tf.cast(tf.shape(image)[1], tf.float32)
    k = np.random.randint(0, 3)
    image = tf.image.rot90(image, k)
    xmin, ymin, xmax, ymax, label = tf.unstack(gtboxes_and_label, axis=1)
    if k == 1:
        xmin, ymin, xmax, ymax = ymin, w-xmax, ymax, w-xmin
    if k == 2:
        xmin, ymin, xmax, ymax = w-xmax, h-ymax, w-xmin, h-ymin
    if k == 3:
        xmin, ymin, xmax, ymax = h-ymax, xmin, h-ymin, xmax

    return image, tf.stack([xmin, ymin, xmax, ymax, label], axis=1)


def random_flip_left_right(img_tensor, gtboxes_and_label):
    def flip_left_right(img_tensor, gtboxes_and_label):
        h, w = tf.cast(tf.shape(img_tensor)[0], tf.float32), tf.cast(tf.shape(img_tensor)[1], tf.float32)
        img_tensor = tf.image.flip_left_right(img_tensor)

        xmin, ymin, xmax, ymax, label = tf.unstack(gtboxes_and_label, axis=1)
        xmin, ymin, xmax, ymax = w-xmax, ymin, w-xmin, ymax
        return img_tensor, tf.stack([xmin, ymin, xmax, ymax, label], axis=1)

    img_tensor, gtboxes_and_label = tf.cond(tf.less(tf.random_uniform(shape=[], minval=0, maxval=1), 0.5),
                                            lambda: flip_left_right(img_tensor, gtboxes_and_label),
                                            lambda: (img_tensor, gtboxes_and_label))
    return img_tensor,  gtboxes_and_label

def random_flip_up_down(img_tensor, gtboxes_and_label):
    def flip_up_down(img_tensor, gtboxes_and_label):
        h, w = tf.cast(tf.shape(img_tensor)[0], tf.float32), tf.cast(tf.shape(img_tensor)[1], tf.float32)
        img_tensor = tf.image.flip_up_down(img_tensor)

        xmin, ymin, xmax, ymax, label = tf.unstack(gtboxes_and_label, axis=1)
        xmin, ymin, xmax, ymax = xmin, h-ymax, xmax, h-ymin
        return img_tensor, tf.stack([xmin, ymin, xmax, ymax, label], axis=1)

    img_tensor, gtboxes_and_label = tf.cond(tf.less(tf.random_uniform(shape=[], minval=0, maxval=1), 0.5),
                                            lambda: flip_up_down(img_tensor, gtboxes_and_label),
                                            lambda: (img_tensor, gtboxes_and_label))
    return img_tensor,  gtboxes_and_label

def random_distort_color(image):
    """
    """
    image = tf.image.random_brightness(image, max_delta=20./255.)
    image = tf.image.random_saturation(image, lower=0.7, upper=1.3)
    image = tf.image.random_hue(image, max_delta=0.2)
    return image

def random_crop(image, gtboxes_and_label, min_object_covered=0.8, aspect_ratio_range=[0.8, 1.2], area_range=[0.5, 1.0]):
    """
    Params:
        image: image
        gtboxes_and_label:
        min_object_covered: minimum gtboxes leaved
        aspect_ratio_range: crop box aspect ratio range
        area_range: max iou of croped image and origin image
    Returns:
        image:
        gtboxes_and_label:
    """
    h, w = tf.cast(tf.shape(image)[0], tf.float32), tf.cast(tf.shape(image)[1], tf.float32)
    xmin, ymin, xmax, ymax, label = tf.unstack(gtboxes_and_label, axis=1)
    bboxes = tf.stack([ ymin/h, xmin/w, ymax/h, xmax/w], axis=1)
    bboxes = tf.clip_by_value(bboxes, 0, 1)
    begin, size, dist_boxes = tf.image.sample_distorted_bounding_box(
                                    tf.shape(image),
                                    bounding_boxes=tf.expand_dims(bboxes, axis=0),
                                    min_object_covered=min_object_covered,
                                    aspect_ratio_range=aspect_ratio_range,
                                    area_range=area_range)
    # NOTE dist_boxes with shape: [ymin, xmin, ymax, xmax] and in values in range(0, 1)
    # Employ the bounding box to distort the image.
    croped_box = [dist_boxes[0,0,1]*w, dist_boxes[0,0,0]*h, dist_boxes[0,0,3]*w, dist_boxes[0,0,2]*h]
    croped_xmin = tf.clip_by_value(xmin, croped_box[0], croped_box[2])-croped_box[0]
    croped_ymin = tf.clip_by_value(ymin, croped_box[1], croped_box[3])-croped_box[1]
    croped_xmax = tf.clip_by_value(xmax, croped_box[0], croped_box[2])-croped_box[0]
    croped_ymax = tf.clip_by_value(ymax, croped_box[1], croped_box[3])-croped_box[1]
    croped_image = tf.slice(image, begin, size)
    
    # filter overlaps < shreshold crop ops
    box_w, box_h = xmax-xmin, ymax-ymin
    croped_box_w, croped_box_h = croped_xmax-croped_xmin, croped_ymax-croped_ymin
    iou = (croped_box_w*croped_box_h)/(box_w*box_h)
    image, boxes = tf.cond(tf.less(tf.reduce_min(iou), min_object_covered),
                            lambda: (image, gtboxes_and_label),
                            lambda: (croped_image, tf.stack([croped_xmin, croped_ymin, croped_xmax, croped_ymax, label], axis=1)),
                          )
    return image, boxes

def random_blur(image):
    def gaussian_blur(image):
        return cv2.GaussianBlur(image, (15,15), 0)
    image = tf.cond(tf.less(tf.random_uniform(shape=[], minval=0, maxval=1), 0.5),
                                            lambda: tf.py_func(gaussian_blur, [image], tf.float32),
                                            lambda: image)
    return image

def random_salt(image, percetage=0.1):
    def pepperand_salt(image, percetage):
        noise_rate = np.random.uniform(0, percetage)
        NoiseNum=int(noise_rate*image.shape[0]*image.shape[1])
        for i in range(NoiseNum):
            randX              = np.random.randint(0, image.shape[0]-1)
            randY              = np.random.randint(0, image.shape[1]-1)
            image[randX,randY] = np.random.randint(0, 255, [3])
        return image

    image = tf.cond(tf.less(tf.random_uniform(shape=[], minval=0, maxval=1), 0.5),
                                            lambda: tf.py_func(pepperand_salt, [image, percetage], tf.float32),
                                            lambda: image)
    return image

def resize_image_correct_bbox(image, boxes, image_h, image_w):
    h, w = tf.cast(tf.shape(image)[0], tf.float32), tf.cast(tf.shape(image)[1], tf.float32)
    image = tf.image.resize_images(image, size=[image_h, image_w])

    # correct bbox
    xx1 = boxes[:, 0] * image_w / w
    yy1 = boxes[:, 1] * image_h / h
    xx2 = boxes[:, 2] * image_w / w
    yy2 = boxes[:, 3] * image_h / h
    idx = boxes[:, 4]

    boxes = tf.stack([xx1, yy1, xx2, yy2, idx], axis=1)
    return image, boxes


