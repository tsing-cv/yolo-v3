#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : dataset.py
#   Author      : YunYang1994, tsing-cv
#   Created date: 2019-01-19 22:43:26
#   Description :
#
#================================================================
# ******************show tfrecord********************************
import sys 
sys.path.append("../../")
# ***************************************************************

import numpy as np
import tensorflow as tf
from core.data_preparation import data_augment

class Parser(object):
    """
    step 1: parse example
    step 2: data augmentation
    step 3: encode gt_boxes
    """
    def __init__(self, image_h, image_w, anchors, num_classes, debug=False):
        self.anchors     = anchors
        self.num_classes = num_classes
        self.image_h     = image_h
        self.image_w     = image_w
        self.debug       = debug

    def parser_example(self, serialized_example):
        features = tf.parse_single_example( serialized_example,
                                            features = {
                                                'image' : tf.FixedLenFeature([], dtype = tf.string),
                                                'boxes' : tf.FixedLenFeature([], dtype = tf.string),
                                                }
                                            )

        image = tf.image.decode_jpeg(features['image'], channels = 3)
        image = tf.image.convert_image_dtype(image, tf.uint8)

        gt_boxes = tf.decode_raw(features['boxes'], tf.float32)
        gt_boxes = tf.reshape(gt_boxes, shape=[-1, 5])
        return image, gt_boxes

    def data_augment(self, image, boxes, output_h, output_w):
        image, boxes = data_augment.random_rot90(image, boxes)
        image, boxes = data_augment.random_flip_left_right(image, boxes)
        image, boxes = data_augment.random_flip_up_down(image, boxes)
        image, boxes = data_augment.random_crop(image, boxes)
        image, boxes = data_augment.resize_image_correct_bbox(image, boxes, output_h, output_w)
        image        = data_augment.random_blur(image)
        image        = data_augment.random_salt(image)
        image        = data_augment.random_distort_color(image)
        return image, boxes

    def encode_gtboxes(self, gt_boxes):
        """
        Preprocess true boxes to training input format
        Parameters:
        -----------
        :param true_boxes: numpy.ndarray of shape [T, 4]
                            T: the number of boxes in each image.
                            4: coordinate => x_min, y_min, x_max, y_max
        :param true_labels: class id
        :param input_shape: the shape of input image to the yolov3 network, [416, 416]
        :param anchors: array, shape=[9,2], 9: the number of anchors, 2: width, height
        :param num_classes: integer, for coco dataset, it is 80
        Returns:
        ----------
        y_true: list(3 array), shape like yolo_outputs, [13, 13, 3, 85]
                            13:cell szie, 3:number of anchors
                            85: box_centers, box_sizes, confidence, probability
        """
        num_layers = len(self.anchors) // 3
        anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]
        grid_sizes = [[self.image_h//x, self.image_w//x] for x in (32, 16, 8)]

        box_centers = (gt_boxes[:, 0:2] + gt_boxes[:, 2:4]) / 2 # the center of box
        box_sizes =    gt_boxes[:, 2:4] - gt_boxes[:, 0:2] # the height and width of box

        gt_boxes[:, 0:2] = box_centers
        gt_boxes[:, 2:4] = box_sizes

        y_true_13 = np.zeros(shape=[grid_sizes[0][0], grid_sizes[0][1], 3, 5+self.num_classes], dtype=np.float32)
        y_true_26 = np.zeros(shape=[grid_sizes[1][0], grid_sizes[1][1], 3, 5+self.num_classes], dtype=np.float32)
        y_true_52 = np.zeros(shape=[grid_sizes[2][0], grid_sizes[2][1], 3, 5+self.num_classes], dtype=np.float32)

        y_true = [y_true_13, y_true_26, y_true_52]
        anchors_max =  self.anchors / 2.
        anchors_min = -anchors_max
        valid_mask = box_sizes[:, 0] > 0

        # Discard zero rows.
        wh = box_sizes[valid_mask]
        # set the center of all boxes as the origin of their coordinates
        # and correct their coordinates
        wh = np.expand_dims(wh, -2)
        boxes_max = wh / 2.
        boxes_min = -boxes_max

        intersect_mins = np.maximum(boxes_min, anchors_min)
        intersect_maxs = np.minimum(boxes_max, anchors_max)
        intersect_wh   = np.maximum(intersect_maxs - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area       = wh[..., 0] * wh[..., 1]

        anchor_area = self.anchors[:, 0] * self.anchors[:, 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)
        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n not in anchor_mask[l]: continue

                i = np.floor(gt_boxes[t,0]/self.image_w*grid_sizes[l][1]).astype('int32')
                j = np.floor(gt_boxes[t,1]/self.image_h*grid_sizes[l][0]).astype('int32')

                k = anchor_mask[l].index(n)
                c = gt_boxes[t, 4].astype('int32')

                y_true[l][j, i, k, 0:4] = gt_boxes[t, 0:4]
                y_true[l][j, i, k,   4] = 1.
                y_true[l][j, i, k, 5+c] = 1.

        return y_true_13, y_true_26, y_true_52

    def pipeline(self, serialized_example):
        image, gt_boxes = self.parser_example(serialized_example)
        image, gt_boxes = self.data_augment(image, gt_boxes, self.image_h, self.image_w)
        if self.debug: return image, gt_boxes

        y_true_13, y_true_26, y_true_52 = tf.py_func(self.encode_gtboxes, inp=[gt_boxes],
                                                     Tout = [tf.float32, tf.float32, tf.float32])
        image = image / 255.
        return image, y_true_13, y_true_26, y_true_52


class dataset(object):
    def __init__(self, parser, tfrecords_path, batch_size, shuffle=None, repeat=True):
        self.parser = parser
        self.filenames = tf.gfile.Glob(tfrecords_path)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.repeat  = repeat
        self._buildup()

    def _buildup(self):
        try:
            TFRecordDataset = tf.data.TFRecordDataset(self.filenames)
        except:
            raise NotImplementedError("No tfrecords found!")

        TFRecordDataset = TFRecordDataset.map(map_func = self.parser.pipeline,
                                              num_parallel_calls = 10)
        TFRecordDataset = TFRecordDataset.repeat() if self.repeat else TFRecordDataset

        if self.shuffle is not None:
            TFRecordDataset = TFRecordDataset.shuffle(self.shuffle)

        TFRecordDataset = TFRecordDataset.batch(self.batch_size).prefetch(self.batch_size)
        self._iterator = TFRecordDataset.make_one_shot_iterator()

    def get_next(self):
        return self._iterator.get_next()


if __name__ == "__main__":
    import scipy.misc as scm
    import cv2
    from core.utils import visual_tools
    from config import cfgs
    sess = tf.Session()

    parser   = Parser(cfgs.IMAGE_H, cfgs.IMAGE_W, np.array(cfgs.ANCHORS), cfgs.NUM_CLASSES, debug=True)
    trainset = dataset(parser, cfgs.train_tfrecord, 1, shuffle=1)

    is_training = tf.placeholder(tf.bool)
    example = trainset.get_next()

    # for l in range(30):
    #     image, boxes = sess.run(example)
    #     image, boxes = image[0], boxes[0]
    #     # print (image)
    #     image = visual_tools.draw_box_cv(image, boxes[:, :4], labels=boxes[:, 4],
    #                                      directions=None, scores=None,
    #                                      plus_rgb_mean=False, is_bgr_channel=False)
    #     cv2.imshow('1' ,np.cast['uint8'](image))
    #     cv2.waitKey(5000)

    images, *y_true = example
    for i in range(10):
        ims, ys = sess.run([images, y_true])
        print (ys, '\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
