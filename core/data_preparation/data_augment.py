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
    """对3通道图像正常，4通道图像会出错，自行先reshape之
    """
    image = tf.image.random_brightness(image, max_delta=32./255.)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.image.random_hue(image, max_delta=0.2)
    return image


def resize_image_correct_bbox(image, boxes, image_h, image_w):

    origin_image_size = tf.to_float(tf.shape(image)[0:2])
    image = tf.image.resize_images(image, size=[image_h, image_w])

    # correct bbox
    xx1 = boxes[:, 0] * image_w / origin_image_size[1]
    yy1 = boxes[:, 1] * image_h / origin_image_size[0]
    xx2 = boxes[:, 2] * image_w / origin_image_size[1]
    yy2 = boxes[:, 3] * image_h / origin_image_size[0]
    idx = boxes[:, 4]

    boxes = tf.stack([xx1, yy1, xx2, yy2, idx], axis=1)
    return image, boxes

def random_crop(image, gtboxes_and_label):
    """
    TODO 2019-02-16 22:21 tsing
    :param image:
    :param gtboxes_and_label:
    :return:
    """
    h, w = tf.cast(tf.shape(image)[0], tf.float32), tf.cast(tf.shape(image)[1], tf.float32)
    xmin, ymin, xmax, ymax, label = tf.unstack(gtboxes_and_label, axis=1)
    bboxes = tf.stack([xmin/w, ymin/h, xmax/w, ymax/h], axis=1)
    bboxes = tf.clip_by_value(bboxes, 0, 1)
    begin, size, dist_boxes = tf.image.sample_distorted_bounding_box(
                                    tf.shape(image),
                                    bounding_boxes=tf.expand_dims(bboxes, axis=0),
                                    min_object_covered=0.1,
                                    aspect_ratio_range=[0.8, 1.2],
                                    area_range=[0.7, 1.0])

    # Employ the bounding box to distort the image.
    image = tf.slice(image, begin, size)
    return image, tf.stack([dist_boxes[0, 0]*(w,h,w,h), tf.expand_dims(label, axis=0)], axis=1)


def blur(image):
    pass

def salt(image):
    pass