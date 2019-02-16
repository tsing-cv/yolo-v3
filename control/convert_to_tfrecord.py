#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2018 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : convert_to_tfrecord.py
#   Author      : tsing-cv
#   Created date: 2019-01-14 14:54:23
#   Description :
#
#================================================================

# NOTE Example of dataset
# ---------------------------------------------------------------
# datesetname
#       JPEG : .jpg or .png
#       ANNOT: .txt 
#             each line is a gtbox: xmin,ymin,xmax,ymax,class,...
#             eg                  : 0, 0, 100, 100, dog
# ---------------------------------------------------------------
import os
import io
import sys
sys.path.append('..')
import numpy as np
import tensorflow as tf
from config import cfgs
from tqdm import tqdm
import scipy.misc as scm

class Convert_To_Tfrecord():
    def __init__(self, dataset_dir, is_train=True):
        self.dataset_dir  = dataset_dir
        self.is_train     = is_train
        self.convert_to_tfrecord()

    @staticmethod
    def int64_feature(value):
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    @staticmethod
    def float_feature(value):
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    @staticmethod
    def bytes_feature(value):
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


    def image_and_gtbox_label(self, image_name_with_format, img_dir, gt_dir):
        """Read the annotation of the image

        Args:
            Str image_name_with_format: image name without format
            List shape: [h, w]
        Returns:
            List gtbox_label: annotations of image
                [[xmin, ymin, xmax, ymax, cls1], [...]]
        """
        image_name,_ = os.path.splitext(image_name_with_format)
        image_data = tf.gfile.GFile(os.path.join(img_dir, image_name_with_format), 'rb').read()
        array_image = scm.imread(io.BytesIO(image_data))
        # scm.imshow(array_image)
        # h, w = array_image.shape[: 2]
        gtbox_labels = []
        txt_path = os.path.join(gt_dir, "{}.txt".format(image_name)) 
        with open(txt_path, 'r', encoding='utf-8') as annot:
            for line in annot:
                line = line.strip().split(',')
                box,label = list(map(float, line[: 4])), cfgs.label_num_map[line[4]]
                gtbox_labels.append(box+[label])
        return image_name, image_data, np.array(gtbox_labels, dtype=np.float32)

    def convert_to_tfrecord(self):
        print ("\n\n==> dataset dir is:{}".format(os.path.abspath(self.dataset_dir)))
        img_dir = os.path.join(self.dataset_dir, 'JPEG')
        assert tf.gfile.Exists(img_dir), "Image folder {} are not exits;Or your images' folder name is not JPEG".format(img_dir)
        gt_dir = os.path.join(self.dataset_dir, 'ANNOT')
        assert tf.gfile.Exists(gt_dir), "Annotations folder {} are not exits;Or your annotations' folder name is not ANNOT".format(gt_dir)
        
        images = tf.gfile.ListDirectory(img_dir)

        if self.is_train:
            tfrecord_path = cfgs.train_tfrecord
        else:
            tfrecord_path = cfgs.test_tfrecord
        if not tf.gfile.Exists(os.path.splitext(tfrecord_path)[0]):
            tf.gfile.MakeDirs(os.path.splitext(tfrecord_path)[0])
        print ('{}\nTfrecord Saving Path:\n\t{}\n{}'.format("@@"*30, tfrecord_path, "@@"*30))
        
        # with tf.io.TFRecordWriter(tfrecord_path) as tfrecord_writer: # NOTE tf1.12+
        with tf.python_io.TFRecordWriter(tfrecord_path) as tfrecord_writer:
            pbar = tqdm(images, total=len(images), ncols=80)
            for image_name_with_format in pbar:
                image_name, image_data, gtbox_label = self.image_and_gtbox_label(image_name_with_format, img_dir, gt_dir)
                pbar.set_description(">>{}".format(image_name))
                example = tf.train.Example(features = tf.train.Features(feature={
                                                    "image_name": self.bytes_feature(image_name.encode()), 
                                                    "image": self.bytes_feature(image_data), 
                                                    "boxes": self.bytes_feature(gtbox_label.tostring()), 
                                                    }))
                tfrecord_writer.write(example.SerializeToString())
        print ("Convert down ! {}".format(tfrecord_path))

if __name__ == "__main__":
    Convert_To_Tfrecord(dataset_dir="{}/dataset/train".format(cfgs.ROOT_DIR),
                        is_train=True)
    Convert_To_Tfrecord(dataset_dir="{}/dataset/val".format(cfgs.ROOT_DIR),
                        is_train=False)